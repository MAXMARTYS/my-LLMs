import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

class sLSTMcell(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden

        self.W_z = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_i = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_f = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=True)

        self.R_z = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_i = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_f = nn.Linear(d_hidden, d_hidden, bias=False)
        self.R_o = nn.Linear(d_hidden, d_hidden, bias=False)

    def step(self, x, h, c, n, m, epsilon=1e-8): # Input, hidden state, cell state, normalizer state

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

class mLSTMcell:
    pass

class mLSTMblock:
    pass

class xLSTM:
    pass

if __name__=='__main__':
    print('Everything works!')