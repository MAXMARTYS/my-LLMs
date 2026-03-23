import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

from .utils import TokenEmbedding
from .lstm_blocks import sLSTMblock, mLSTMblock

class xLSTM(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, block_types):
        super().__init__()
        self.embedding = TokenEmbedding(pretrained=False, embed_dim=d_model)
        vocab_size, embed_dim = self.embedding.get_info()

        assert set(block_types).issubset({'s', 'm'}), 'block_types must only contain "s" (sLSTM) or "m" (mLSTM)'

        self.blocks = nn.ModuleList([
            mLSTMblock(d_model) if block == 'm' else sLSTMblock(d_model, n_heads) for block in block_types
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.embed.weight

    def forward(self, x, states=None):
        x = self.embedding(x)
 
        if states is None:
            states = [None] * len(self.blocks)
 
        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block(x, state)
            new_states.append(new_state)
 
        x = self.norm(x)
        out = self.head(x)
 
        # return out, new_states
        return out # For now I will just return the output

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
 
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
 
            idx = torch.cat([idx, idx_next], dim=1)
 
        return idx
 

if __name__ == '__main__':
    model = xLSTM(d_model=512, d_hidden=2048, n_heads=4, block_types=['s', 's', 's', 'm', 's', 's', 's', 'm'])
    summary = summary(
        model, 
        input_size=(2, 512),  # (batch_size, seq_len)
        dtypes=[torch.long],  # Specify integer type
    )
    print('Everything works!')