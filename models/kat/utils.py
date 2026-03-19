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

# Old implementation without grouping, which is too expensive to compute  
# class GroupedRationalKANLinear(nn.Module):
#     def __init__(self, in_shape, out_shape, m=4, n=3):
#         super().__init__()
#         # Rational KANs use rational activation functions of P(x) / (1+|Q(x)|) for activation 
#         assert m <= 6 and n <= 6, 'Polynomials must be of degree below 7' # Avoid too much complexity

#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.m = m
#         self.n = n

#         # There are some possible improvements we can make
#         # One idea: represent the polynomial as a0 + x*(a1 + x*(a2 + x(a3 + x*(...)))) instead of a0 + a1*x + a2*x^2 + a3*x^3 + ...
#         self.P = nn.Parameter(torch.randn(out_shape, in_shape, m + 1) * 0.1)
#         self.Q = nn.Parameter(torch.randn(out_shape, in_shape, n + 1) * 0.1)

#     def forward(self, x):
        
#         powers = torch.stack([x**i for i in range(self.degree+1)], dim=-1)
#         powers = powers.unsqueeze(1)

#         P_x = (powers * self.P).sum(dim=-1)
#         Q_x = (powers * self.Q).sum(dim=-1)

#         phi = P_x / (1 + Q_x.abs())

#         return phi.sum(dim=-1)
    
class GroupedRationalKANLinear(nn.Module):
    def __init__(self, in_shape, out_shape, m=4, n=3, groups=8):
        super().__init__()
        assert m <= 6 and n <= 6, 'Polynomials must be of degree below 7'
        assert in_shape % groups == 0, 'in_shape must be divisible by groups'

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.m = m
        self.n = n
        self.groups = groups
        self.group_size = in_shape // groups

        self.P = nn.Parameter(torch.randn(groups, self.group_size, m + 1) * 0.1)
        self.Q = nn.Parameter(torch.randn(groups, self.group_size, n + 1) * 0.1)
        nn.init.ones_(self.P[..., 1])

        self.linear = nn.Linear(in_shape, out_shape)

    def forward(self, x):
        leading = x.shape[:-1]

        x_grouped = x.view(*leading, self.groups, self.group_size)

        # There are some possible improvements we can make
        # powers_P = torch.stack([x_grouped ** i for i in range(self.m + 1)], dim=-1)
        # powers_Q = torch.stack([x_grouped ** i for i in range(self.n + 1)], dim=-1)

        # P_x = (powers_P * self.P).sum(dim=-1)
        # Q_x = (powers_Q * self.Q).sum(dim=-1)

        # Horer's method: represent the polynomial as a0 + x*(a1 + x*(a2 + x(a3 + x*(...)))) instead of a0 + a1*x + a2*x^2 + a3*x^3 + ...
        P_x = self.P[..., self.m].clone().expand(*leading, self.groups, self.group_size)
        for i in range(self.m - 1, -1, -1):
            P_x = P_x * x_grouped + self.P[..., i]

        Q_x = self.Q[..., self.n].clone().expand(*leading, self.groups, self.group_size)
        for i in range(self.n - 1, -1, -1):
            Q_x = Q_x * x_grouped + self.Q[..., i]

        phi = P_x / (1 + Q_x.abs())
        phi = phi.view(*leading, self.in_shape)

        out = self.linear(phi)

        return out

class RationalKANAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, m=4, n=3, groups=8):
        super().__init__()
        assert embed_dim % num_heads == 0, 'd_model must be divisible by num_heads'

        self.num_heads = num_heads
        self.h_dim = embed_dim // num_heads

        self.W_k = GroupedRationalKANLinear(embed_dim, embed_dim, m=m, n=n, groups=groups) 
        self.W_q = GroupedRationalKANLinear(embed_dim, embed_dim, m=m, n=n, groups=groups) 
        self.W_v = GroupedRationalKANLinear(embed_dim, embed_dim, m=m, n=n, groups=groups)
        self.out = GroupedRationalKANLinear(embed_dim, embed_dim, m=m, n=n, groups=groups)

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

        output, _ = RationalKANAttention.scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(B, T, C) # Combining multiple heads
        output = self.out(output)

        return output

class KATBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, m=4, n=3, groups=8, ffn_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RationalKANAttention(embed_dim, num_heads, m=4, n=3, groups=8)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            GroupedRationalKANLinear(embed_dim, ffn_ratio * embed_dim, m=m, n=n, groups=groups),
            nn.Dropout(0.2),
            GroupedRationalKANLinear(ffn_ratio * embed_dim, embed_dim, m=m, n=n, groups=groups),
            nn.Dropout(0.2)
        )
        self.resid_dropout1 = nn.Dropout(0.2)
        self.resid_dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x + self.resid_dropout1( self.attn( self.norm1(x) ))
        x = x + self.resid_dropout2( self.ffn( self.norm2(x) ))
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