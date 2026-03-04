import torch
import torch.nn as nn

from tqdm import tqdm

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