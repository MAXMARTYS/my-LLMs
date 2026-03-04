import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

from utils import TokenEmbedding
from lstm_blocks import sLSTMblock, mLSTMblock

# Still needs some debugging
class xLSTM:
    def __init__(self, d_model, d_hidden, n_heads, block_types):
        self.embedding = TokenEmbedding(pretrained=False, embed_dim=d_model)
        vocab_size, embed_dim = self.embedding.get_info() 

        assert set(block_types) == set(['s', 'm'])

        self.blocks = nn.ModuleList([
            mLSTMblock(d_model) if block == 'm' else sLSTMblock(d_model, n_heads) for block in block_types
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.embed.weight

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        out = self.head(x)
        return out

    @torch.no_grad()
    def generate():
        pass

if __name__=='__main__':
    slstm = sLSTMblock(d_hidden=128, n_heads=4)
    dummy_x = torch.randn(2, 10, 128)
    out, state = slstm(dummy_x)
    print('sLSTM output shape:', out.shape)
    summary(
        slstm, 
        input_size=dummy_x.shape,  # (batch_size, seq_len)
        dtypes=[torch.float],  # Specify integer type
    )
    print('Everything works!')