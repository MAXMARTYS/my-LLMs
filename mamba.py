import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

# TODO: 
# - think about using CUDA kernel tricks to speed up the training process

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
        vocab_size, d_model = self.embed.weight.shape
        return vocab_size, d_model

    def forward(self, x):
        return self.embed(x)
    
class SSM(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # self.A = nn.Linear(d_model, d_hidden) # Hidden value updater
        self.A = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Linear(d_model, d_hidden) # Input - hidden state translation 
        self.C = nn.Linear(d_hidden, d_model) # Hidden state - output translation
        self.D = nn.Linear(d_model, d_model) # Skip connection
        self.delta = nn.Linear(d_model, d_hidden)

    def forward(self, x):

        delta = F.softplus( self.delta(x) )

        # A_t = self.A(x)
        A_bar = torch.exp( delta * self.A )

        # B_x = self.B(x)
        B_bar = delta * self.B(x)

        # New implementation - pytorch vectorization TODO: this may have an error
        A_cum = torch.cumprod(A_bar, dim=1)
        # h = torch.cumsum(B_x * A_cum, dim=1)
        h = torch.cumsum(B_bar / A_cum, dim=1) * A_cum

        out = self.C(h) + self.D(x)
        return out

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.linear1 = nn.Linear(d_model, d_model, bias=False)
        self.linear2 = nn.Linear(d_model, d_model, bias=False)
        self.last_linear = nn.Linear(d_model, d_model, bias=False)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.act = nn.SiLU()

        self.ssm = SSM(d_model, d_hidden)

    def forward(self, x):
        x = self.linear1(x)

        x = self.conv( x.transpose(1, 2) ).transpose(1, 2)
        x = self.act(x)
        x = self.ssm(x)

        x_res = self.linear2(x)
        x_res = self.act(x_res)

        z = x * x_res
        out = self.last_linear(z)
        out = x + out # Residual connection
        return out

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