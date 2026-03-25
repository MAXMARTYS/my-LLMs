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
    
    def _init_states(self, batch_size, device):
        states = []
        for block in self.blocks:
            kernel_size = block.conv.kernel_size
            conv_buf = torch.zeros(batch_size, block.d_model, kernel_size - 1, device=device)
            h_ssm = torch.zeros(batch_size, block.ssm.d_hidden, device=device)
            states.append((conv_buf, h_ssm))
        return states
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        B, T = idx.shape
        device = idx.device

        states = self._init_states(B, device)

        for t in range(T):
            x_t = self.embedding(idx[:, t])
            new_states = []
            for i, block in enumerate(self.blocks):
                x_t, new_state = block.step(x_t, states[i])
                new_states.append(new_state)
            states = new_states

        for _ in range(max_new_tokens):
            logits = self.head(self.norm(x_t))
            logits = logits / max(temperature, 1e-5)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            x_t = self.embedding(next_token.squeeze(-1))
            new_states = []
            for i, block in enumerate(self.blocks):
                x_t, new_state = block.step(x_t, states[i])
                new_states.append(new_state)
            states = new_states

        return idx


if __name__=='__main__':
    # Check if there are no initialization errors
    model = MambaModel(d_model=512, d_hidden=2048, n_blocks=6)
    summary(
        model, 
        input_size=(2, 512),  # (batch_size, seq_len)
        dtypes=[torch.long],  # Specify integer type
    )
    print('Everything works!')