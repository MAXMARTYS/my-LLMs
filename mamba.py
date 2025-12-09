import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

torch.scan()

class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        gpt2 = AutoModel.from_pretrained('gpt2')
        embedding = gpt2.get_input_embeddings().weight.detach().clone()
        vocab_size, d_model = embedding.shape

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = nn.Parameter(embedding)

        # Freeze the embedding
        self.embed.weight.requires_grad = False 

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

        self.A = nn.Parameter(torch.randn(d_hidden)) # Hidden value updater
        self.B = nn.Linear(d_model, d_hidden) # Input - hidden state translation 
        self.C = nn.Linear(d_hidden, d_model) # Hidden state - output translation
        self.D = nn.Linear(d_model, d_hidden) # Skip connection
        self.delta = nn.Linear(d_model, d_hidden)

    def forward(self, x):
        B, T, D = x.size() # Batch size, tokens (sequence length), dimentions (embedding)

        x_delta = F.softplus( self.delta(x) )
        A_bar = torch.exp( -x_delta * self.A )
        B_x = self.B(x)

        h = torch.zeros(B, self.d_hidden, device=x.device)
        outputs = []    

        for t in range(T):
            h = A_bar[:, t] * h + B_x[:, t]
            outputs.append(h)

        h = torch.stuck(outputs, dim=1)

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

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.act = nn.SiLU()

        self.ssm = SSM(d_model)

    def forward(self, x):
        x = self.linear1(x)

        x = self.conv(x)
        x = self.act(x)
        x = self.ssm(x)

        x_res = self.linear2(x)
        x_res = self.act(x_res)

        z = x @ x_res
        out = self.last_linear(z)
        return out

