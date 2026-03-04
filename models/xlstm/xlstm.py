import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchinfo import summary

from lstm_blocks import sLSTMblock, mLSTMblock

class xLSTM:
    def __init__(self):
        pass 

    def forward():
        pass 

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