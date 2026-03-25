import torch
import torch.nn as nn

from utils import BlockDiagonal, CausalConv1D, Swish

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
        # h_next = o * ( c_next / (n_next + epsilon) )
        h_next = o * (c_next / torch.clamp(torch.abs(n_next), min=epsilon))

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

        x_conv = x_norm.transpose(1, 2)
        x_if = self.swish( self.conv(x_conv).transpose(1, 2) ) # Input for i and f gates
        x_if = self.dropout(x_if)

        x_zo = x_norm # Input for z and o gates

        hs = []
        for t in range(T):
            h, c, n, m = self.step(x_if[:, t], x_zo[:, t], h, c, n, m)
            hs.append(h)

        h_seq = torch.stack(hs, dim=1)

        h_gn = self.gn(h_seq)
        h_gn = self.dropout(h_gn)
        h_skip = h_gn + x # First skip connection
        h_skip_norm = self.ln(h_skip)

        z_left = self.left_linear(h_skip_norm)
        z_right = self.gelu( self.right_linear(h_skip_norm) )

        z = z_left * z_right
        z = self.dropout(z)

        out = self.last_linear(z)
        out = self.dropout(out)
        out = out + h_skip # Second skip connection

        return out, (h, c, n, m)

class mLSTMblock(nn.Module):
    def __init__(self, d_hidden, n_heads=4):
        super().__init__()
        self.d_hidden = d_hidden

        self.W_i = nn.Linear(d_hidden*2, 1, bias=True)
        self.W_f = nn.Linear(d_hidden*2, 1, bias=True)
        self.W_o = nn.Linear(d_hidden*2, d_hidden*2, bias=True)

        # self.W_q = nn.Linear(d_hidden*2, d_hidden, bias=True)
        # self.W_k = nn.Linear(d_hidden*2, d_hidden, bias=True)
        # self.W_v = nn.Linear(d_hidden*2, d_hidden, bias=True)

        self.W_q = BlockDiagonal(d_hidden*2, d_hidden*2, n_heads, bias=False)
        self.W_k = BlockDiagonal(d_hidden*2, d_hidden*2, n_heads, bias=False)
        self.W_v = BlockDiagonal(d_hidden*2, d_hidden*2, n_heads, bias=False)

        self.conv = CausalConv1D(self.d_hidden*2, self.d_hidden*2, kernel_size=4)

        self.skip_proj = nn.Linear(d_hidden*2, d_hidden*2, bias=False)

        self.ln = nn.LayerNorm(d_hidden)
        # self.gn = nn.LayerNorm(d_hidden)
        self.gn = nn.GroupNorm(num_groups=n_heads, num_channels=d_hidden*2)

        self.left_linear = nn.Linear(d_hidden, int(d_hidden*2))
        self.right_linear = nn.Linear(d_hidden, int(d_hidden*2))
        self.last_linear = nn.Linear(int(d_hidden*2), d_hidden)

        self.gelu = nn.GELU()
        self.swish = Swish()
        self.dropout = nn.Dropout(0.2)

        self.scale = float(d_hidden) ** -0.5

    def step(self, x, x_trans, c, n, m, epsilon=1e-8):
        
        i_bar = self.W_i(x) # Input gate
        f_bar = self.W_f(x) # Forget gate
        o_bar = self.W_o(x) # Output gate

        q = self.W_q(x_trans) # Query
        # k = (1 / torch.sqrt(self.d_hidden)) * self.W_k(x_trans) # Key
        k = self.scale * self.W_k(x_trans) # Key
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
        cq = cq.squeeze(-1)
        denom = torch.abs((n_next * q).sum(-1, keepdim=True)) 
        denom = torch.maximum(denom, torch.ones_like(denom))
        h_bar = cq / (denom + epsilon)
        h = o * h_bar

        return h, c_next, n_next, m_next


    def forward(self, x, state=None):
        B, T, D = x.shape

        if state is None:
            h = x.new_zeros(B, D*2)
            c = x.new_zeros(B, D*2, D*2)
            n = x.new_zeros(B, D*2)
            m = x.new_zeros(B, 1)
        else:
            h, c, n, m = state

        x_norm = self.ln(x)

        x_left = self.left_linear(x_norm)
        x_right = self.right_linear(x_norm)

        x_conv = x_left.transpose(1, 2)
        x_trans = self.swish( self.conv(x_conv).transpose(1, 2) )

        # This loop is technically vectorizable (like in Mamba) --> TODO: I can try to add it later
        hs = []
        for t in range(T):
            h, c, n, m = self.step(x_left[:, t], x_trans[:, t], c, n, m)
            hs.append(h)

        h_seq = torch.stack(hs, dim=1)

        h_gn = self.gn(h_seq.transpose(1, 2)).transpose(1, 2)
        h_gn = h_gn + self.skip_proj(x_trans)

        x_right = self.swish(x_right)
        h = h_gn * x_right 

        out = self.last_linear(h)
        out = out + x # Second skip connection

        return out, (h, c, n, m)
