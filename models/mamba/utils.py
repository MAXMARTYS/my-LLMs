import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tqdm import tqdm

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

# Causal convolution from xlstm
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        if self.padding <= 0:
            self.padding = 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0)) 
        x = self.conv(x)
        return x

class SSM(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # self.A = nn.Linear(d_model, d_hidden) # Hidden value updater
        # self.A = nn.Parameter(torch.randn(d_hidden))
        self.A = nn.Parameter(torch.log(torch.arange(1, d_hidden + 1).float()))
        self.B = nn.Linear(d_model, d_hidden) # Input - hidden state translation 
        self.C = nn.Linear(d_hidden, d_model) # Hidden state - output translation
        self.D = nn.Linear(d_model, d_model) # Skip connection
        self.delta = nn.Linear(d_model, d_hidden)

    def forward(self, x):
        ''' 
        This is a new implementation that moves the computation into the log-space. This should prevent gradient explosion.
        It works on an assumption that cumprod(exp(A) = exp(cumsum(A))
        '''
        B_seq, T, C = x.shape

        delta = F.softplus( self.delta(x) )
        A_log = -F.softplus(self.A)
        A_bar_log = delta * A_log

        B_bar = delta * self.B(x)

        A_log_cum = torch.cumsum(A_bar_log, dim=1).clamp(min=-20, max=20)

        # Shift A_log_cum right by 1 so B_n is not divided by A_n
        A_log_cum_shifted = torch.roll(A_log_cum, shifts=1, dims=1)
        A_log_cum_shifted[:, 0, :] = 0.0  # first step has no prior A products

        B_shift = B_bar * torch.exp(-A_log_cum_shifted)
        h = torch.exp(A_log_cum) * torch.cumsum(B_shift, dim=1)

        out = self.C(h) + self.D(x)
        return out

    # def forward(self, x):
    #     B_seq, T, _ = x.shape

    #     # There are some issues with gradients - I will clamp the values for now. TODO: investigate this further
    #     delta = F.softplus( self.delta(x) ).clamp(max=10.0) 
    #     A_bar = torch.exp( delta * -F.softplus(self.A) ).clamp(min=1e-6)

    #     B_bar = delta * self.B(x)

    #     # New implementation - pytorch vectorization TODO: this may have an error
    #     # A_cum = torch.cumprod(A_bar, dim=1)
    #     # h = torch.cumsum(B_bar / A_cum, dim=1) * A_cum

    #     # A_cum = torch.cumprod(A_bar, dim=1)
    #     # A_cum_prev = torch.cat([
    #     #     torch.ones(B_seq, 1, self.d_hidden, device=x.device, dtype=x.dtype),
    #     #     A_cum[:, :-1, :]
    #     # ], dim=1)
    #     # B_pre = B_bar * A_cum_prev
    #     # h = A_cum * torch.cumsum(B_pre, dim=1)

    #     # Normal loop implementation - for reference
    #     h = torch.zeros(B_seq, self.d_hidden, device=x.device, dtype=x.dtype)
    #     outs = []
    #     for t in range(T):
    #         h = A_bar[:, t] * h + B_bar[:, t]
    #         outs.append(h)
    #     h = torch.stack(outs, dim=1)

    #     out = self.C(h) + self.D(x)
    #     return out

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)

        self.d_model = d_model
        self.d_hidden = d_hidden

        self.linear1 = nn.Linear(d_model, d_model, bias=False)
        self.linear2 = nn.Linear(d_model, d_model, bias=False)
        self.last_linear = nn.Linear(d_model, d_model, bias=False)

        # self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.conv = CausalConv1D(d_model, d_model, kernel_size=3, groups=d_model)
        self.act = nn.SiLU()

        self.ssm = SSM(d_model, d_hidden)

    def step(self, x, state):
        res = x
        x = self.norm(x)
        conv_state, ssm_state = state 

        x_branch = self.linear1(x)
        z_branch = self.linear2(x)

        x_conv_current = x_branch.unsqueeze(-1) # (B, d_model, 1)
        full_conv_window = torch.cat([conv_state, x_conv_current], dim=-1) # (B, d_model, 3)
        
        x_ssm = (full_conv_window * self.conv.weight.squeeze(1)).sum(dim=-1)
        if self.conv.bias is not None:
            x_ssm = x_ssm + self.conv.bias
            
        x_ssm = self.act(x_ssm)

        ssm_delta = F.softplus(self.ssm.delta(x_ssm))
        ssm_A_log = -F.softplus(self.ssm.A)
        
        A_bar = torch.exp(ssm_delta * ssm_A_log) 
        B_bar = ssm_delta * self.ssm.B(x_ssm)

        ssm_state = A_bar * ssm_state + B_bar
        x_ssm_out = self.ssm.C(ssm_state) + self.ssm.D(x_ssm)

        combined = x_ssm_out * self.act(z_branch)
        out = self.last_linear(combined)
        
        next_conv_state = full_conv_window[:, :, 1:] 
        
        return out + res, (next_conv_state, ssm_state)

    def forward(self, x):
        res = x
        x = self.norm(x)

        # SSM path
        x_ssm = self.linear1(x) 
        x_ssm = self.conv(x_ssm.transpose(1, 2)).transpose(1, 2)
        x_ssm = self.act(x_ssm) 
        x_ssm = self.ssm(x_ssm)

        # Gated path
        z = self.linear2(x)
        z = self.act(z)

        combined = x_ssm * z 
        out = self.last_linear(combined)
        out = out + res
        return out

def calculate_perplexity(model, dataloader, device, max_batches, ignore_index=0):
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    batch_size = next(iter(dataloader))['input_ids'].shape[0]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating perplexity', total=max_batches//batch_size):
            input_ids = batch['input_ids'].to(device)

            inputs  = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()

            logits = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)

            # Count non-padding tokens only
            non_pad = (targets != ignore_index).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

            n_batches += batch_size

            if n_batches >= max_batches:
                break

    avg_nll = total_loss / total_tokens 
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity

# if __name__=='__main__':
#     def check_causality(model, B_seq=2, T=16, C=32, device='cpu'):
#         """
#         Checks if the model peeks into the future.
#         Strategy: perturb input at time step t, check if any output at t' < t changes.
#         """
#         model.eval()
#         with torch.no_grad():
#             x = torch.randn(B_seq, T, C, device=device)
#             out_original = model(x).clone()

#             violations = []

#             for t in range(1, T):  # perturb each timestep (skip 0, nothing before it)
#                 x_perturbed = x.clone()
#                 x_perturbed[:, t, :] += torch.randn(B_seq, C, device=device) * 10.0

#                 out_perturbed = model(x_perturbed)
#                 diff = (out_perturbed - out_original).abs()

#                 # Check if any output at t' < t changed
#                 if diff[:, :t, :].max().item() > 1e-5:
#                     violations.append(t)

#             if violations:
#                 print(f"Model is NOT causal! Future-peeking detected at input timesteps: {violations}")
#             else:
#                 print(f"Model is causal. No future-peeking detected across {T} timesteps.")

#             return len(violations) == 0
#     check_causality(MambaBlock(d_model=32, d_hidden=128))