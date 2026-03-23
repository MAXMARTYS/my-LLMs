import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

from .utils import TokenEmbedding, TransformerBlock

class Transformer(nn.Module):
    def __init__(self, depth, num_heads):
        super().__init__()

        self.embedding = TokenEmbedding(pretrained=False, embed_dim=512)
        vocab_size, embed_dim = self.embedding.get_info()
        # print(vocab_size, embed_dim)

        self.pos_emb = nn.Embedding(2048, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.head.weight = self.embedding.embed.weight
        # self.head.weight.requires_grad = False  # Keep frozen

    def forward(self, idx):
        B, T = idx.shape

        emb = self.embedding(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]

        x = emb + pos

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)

        return x
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 2048 else idx[:, -2048:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] # Get last token
            
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    
if __name__=='__main__':
    # Check if there are no initialization errors
    model = Transformer(num_heads=8, depth=6)
    summary = summary(
        model, 
        input_size=(2, 512),  # (batch_size, seq_len)
        dtypes=[torch.long],  # Specify integer type
    )
    print(summary)
    print(summary.total_params)
    print('Everything works!')