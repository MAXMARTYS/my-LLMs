import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

# TODO: make both cells multiheaded
# I believe a correct way of doing that is Conv1D instead of linear, with groups=n_heads
# Only then, I will be able to move to full blocks
class sLSTMcell(nn.Module):
    def __init__(self, d_hidden, n_heads):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        assert d_hidden % n_heads == 0, 'd_hidden must be divisible by n_heads'
        self.d_head = d_hidden // n_heads

        self.W_z = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_i = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_f = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=True)

        self.R_z = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_i = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_f = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_o = nn.Linear(d_hidden, d_hidden, bias=False)

    def step(self, x, h, c, n, m, epsilon=1e-8): # Input, hidden state, cell state, normalizer state, staabilizer state
        B, T, D = x.shape

        z_bar = self.W_z(x) + self.R_z(h) # Cell input
        i_bar = self.W_i(x) + self.R_i(h) # Input gate
        f_bar = self.W_f(x) + self.R_f(h) # Forget gate
        o_bar = self.W_o(x) + self.R_o(h) # Output gate

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

        hs = []
        for t in range(T):
            h, c, n, m = self.step(x[:, t], h, c, n, m)
            hs.append(h)

        h_seq = torch.stack(hs, dim=1)
        return h_seq, (h, c, n, m)
        
 
class sLSTMblock:
    pass

class mLSTMcell(nn.Module):
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
    
class mLSTMblock:
    pass

class xLSTM:
    pass

if __name__=='__main__':
    slstm = sLSTMcell(d_hidden=128, n_heads=4)
    dummy_x = torch.randn(2, 10, 128)
    out, state = slstm(dummy_x)
    print('sLSTM output shape:', out.shape)
    summary(
        slstm, 
        input_size=dummy_x.shape,  # (batch_size, seq_len)
        dtypes=[torch.float],  # Specify integer type
    )
    print('Everything works!')