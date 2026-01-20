import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

# Code for block diagonal and causal conv taken from this repo:
# https://github.com/styalai/xLSTM-pytorch

# I am still afraid that these modules are wrong with my implementation
class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        assert out_features % num_blocks == 0
        
        block_out_features = out_features // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, block_out_features, bias=bias)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        if self.padding <= 0:
            self.padding = 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class sLSTMblock(nn.Module):
    def __init__(self, d_hidden, n_heads):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        assert d_hidden % n_heads == 0, 'd_hidden must be divisible by n_heads'
        self.d_head = d_hidden // n_heads

        # Block diagonals instead of linear to simulate multiple heads
        self.W_z = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=True)
        self.W_i = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=True)
        self.W_f = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=True)
        self.W_o = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=True)

        self.R_z = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=False)
        self.R_i = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=False)
        self.R_f = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=False)
        self.R_o = BlockDiagonal(d_hidden, d_hidden, n_heads, bias=False)

        self.conv = CausalConv1D(self.d_hidden, self.d_hidden, kernel_size=4)

        self.dropout = nn.Dropout(0.2)

        self.ln = nn.LayerNorm(d_hidden)
        self.gn = nn.LayerNorm(d_hidden)

        self.left_linear = nn.Linear(d_hidden, int(d_hidden*4/3))
        self.right_linear = nn.Linear(d_hidden, int(d_hidden*4/3))

        self.last_linear = nn.Linear(int(d_hidden*4/3), d_hidden)

        self.gelu = nn.GELU()
        self.swish = Swish()

    def step(self, x_if, x_zo, h, c, n, m, epsilon=1e-8): # Input, hidden state, cell state, normalizer state, staabilizer state

        z_bar = self.W_z(x_zo) + self.R_z(h) # Cell input
        i_bar = self.W_i(x_if) + self.R_i(h) # Input gate
        f_bar = self.W_f(x_if) + self.R_f(h) # Forget gate
        o_bar = self.W_o(x_zo) + self.R_o(h) # Output gate

        z = torch.tanh(z_bar)
        # i = torch.exp(i_bar)
        # f = torch.exp(f_bar)
        o = torch.sigmoid(o_bar)

        m_next = torch.maximum( f_bar + m, i_bar ) # We can also use log(f) and log(i)

        i = torch.exp(i_bar - m_next)
        f = torch.exp(f_bar + m - m_next)

        c_next = f * c + i * z
        n_next = f * n + i
        h_next = o * ( c_next / (n_next + epsilon) )

        return h_next, c_next, n_next, m_next
    
    def forward(self, x, state=None):
        B, T, D = x.shape

        if state is None:
            h = x.new_zeros(B, D)
            c = x.new_zeros(B, D)
            n = x.new_zeros(B, D)
            m = x.new_zeros(B, D)
        else:
            h, c, n, m = state

        x_norm = self.ln(x)

        x_if = self.swish( self.conv(x_norm) ) # Input for i and f gates
        x_if = self.dropout(x_if)
        x_zo = x # Input for z and o gates

        hs = []
        for t in range(T):
            h, c, n, m = self.step(x_if[:, t], x_zo[:, t], h, c, n, m)
            hs.append(h)

        h_seq = torch.stack(hs, dim=1)

        h_gn = self.gn(h_seq)
        h_gn = self.drouput(h_gn)
        h_skip = h_gn + x # First skip connection
        h_skip_norm = self.ln(x)

        z_left = self.left_linear(h_skip_norm)
        z_right = self.gelu( self.right_linear(h_skip_norm) )

        z = z_left * z_right
        z = self.drouput(z)

        out = self.last_linear(z)
        out = self.dropout(out)
        out = out + h_skip # Second skip connection

        return out, (h, c, n, m)

class mLSTMblock(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden

        self.W_i = nn.Linear(d_hidden, 1, bias=True)
        self.W_f = nn.Linear(d_hidden, 1, bias=True)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=True)

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=True)

    def step(self, x, c, n, m):
        
        i_bar = self.W_i(x) # Input gate
        f_bar = self.W_f(x) # Forget gate
        o_bar = self.W_o(x) # Output gate

        q = self.W_q(x) # Query
        k = (1 / torch.sqrt(self.d_hidden)) * self.W_k(x) # Key
        v = self.W_v(x) # Value

        m_next = torch.maximum( f_bar + m, i_bar ) # We can also use log(f) and log(i)

        i = torch.exp(i_bar - m_next)
        f = torch.exp(f_bar + m - m_next)
        o = torch.sigmoid(o_bar)

        vkT = v.unsqueeze(-1) @ k.unsqueeze(-2) 
        c_next = f.unsqueeze(-1) * c + i.unsqueeze(-1) * vkT
        n_next = f * n + i * k 

        # Still no idea why we compute the hidden state in mLSTM
        cq = c_next @ q.unsqueeze(-1)
        denominator = torch.maximum(torch.abs(n_next.T * q).sum(-1, keepdim=True), x.new_ones((x.size(0), 1)))
        h_bar = cq / denominator
        h = o * h_bar

        return h, c_next, n_next, m_next


    def forward(self, x, state=None):
        B, T, D = x.shape

        if state is None:
            h = x.new_zeros(B, D)
            c = x.new_zeros(B, D)
            n = x.new_zeros(B, D)
            m = x.new_zeros(B, D)
        else:
            h, c, n, m = state

        # This loop is technically vectorizable (like in Mamba) --> I can try to add it later
        hs = []
        for t in range(T):
            h, c, n, m = self.step(x[:, t], c, n, m)
            hs.append(h)

        h_seq = torch.stack(hs, dim=1)
        return h_seq, (h, c, n, m)

class xLSTM:
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