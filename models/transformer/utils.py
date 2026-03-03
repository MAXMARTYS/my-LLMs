import torch
import torch.nn as nn
from transformers import AutoModel
from tqdm import tqdm

class TokenEmbedding(nn.Module):
    def __init__(self, pretrained=False, vocab_size=50257, embed_dim=None):
        super().__init__()

        if pretrained:
            gpt2 = AutoModel.from_pretrained('gpt2')
            embedding = gpt2.get_input_embeddings().weight.detach().clone()
            self.vocab_size, self.embed_dim = embedding.shape

            self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
            self.embed.weight = nn.Parameter(embedding)

            self.embed.weight.requires_grad = False 

        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim

    def get_info(self):
        return self.vocab_size, self.embed_dim

    def forward(self, x):
        return self.embed(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, 'd_model must be divisible by num_heads'

        self.num_heads = num_heads
        self.h_dim = embed_dim // num_heads

        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False) # Key transformation
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False) # Query transformation
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False) # Value transformation
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(0.2)

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask = None, dropout = None):
        d_k = Q.shape[-1]

        attn_scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        attn_scores = attn_scores.softmax(dim=-1)

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ V), attn_scores

    def forward(self, x):
        B, T, C = x.size() # Batch (size), tokens (sequence length), chanels (embedding dim)

        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        key = self.W_k(x)
        query = self.W_q(x)
        value = self.W_v(x)

        key = key.view(B, T, self.num_heads, self.h_dim).transpose(1, 2)
        query = query.view(B, T, self.num_heads, self.h_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.h_dim).transpose(1, 2)

        output, _ = MultiHeadAttention.scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(B, T, C) # Combining multiple heads
        output = self.out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(0.2)
        )
        self.resid_dropout1 = nn.Dropout(0.2)
        self.resid_dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x + self.resid_dropout1( self.attn( self.norm1(x) ))
        x = x + self.resid_dropout2( self.mlp( self.norm2(x) ))
        return x
    
def calculate_perplexity(model, dataloader, device, max_batches, ignore_index=0):
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    batch_size = next(iter(dataloader))['input_ids'].shape[0]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating perplexity', total=max_batches//batch_size):
            input_ids = batch['input_ids'].to(device)

            inputs  = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()

            logits = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)

            # Count non-padding tokens only
            non_pad = (targets != ignore_index).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

            n_batches += batch_size

            if n_batches >= max_batches:
                break

    avg_nll = total_loss / total_tokens 
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity