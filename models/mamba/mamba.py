import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

from utils import TokenEmbedding, MambaBlock

class MambaModel(nn.Module):
    def __init__(self, d_model, d_hidden, n_blocks):
        super().__init__()
        self.embedding = TokenEmbedding(pretrained=False, embed_dim=d_model)
        vocab_size, embed_dim = self.embedding.get_info()

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_hidden) for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.head.weight = self.embedding.embed.weight
        # self.head.weight.requires_grad = False  # Keep frozen

    def forward(self, x):
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        out = self.head(x)

        return out
    
    @torch.no_grad()
    def generate(self):
        pass


if __name__=='__main__':
    # Check if there are no initialization errors
    model = MambaModel(d_model=512, d_hidden=2048, n_blocks=6)
    summary(
        model, 
        input_size=(2, 512),  # (batch_size, seq_len)
        dtypes=[torch.long],  # Specify integer type
    )
    print('Everything works!')