import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk

from mamba import MambaModel

# Extend sequences to a fixed length
def collate_batch(batch, max_len=512):
    input_ids = [item['input_ids'][:max_len] for item in batch]  # truncate

    input_ids = [
        torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
        if len(seq) < max_len else seq
        for seq in input_ids
    ]

    input_ids = torch.stack(input_ids)

    attention_mask = (input_ids != 0).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

# Load data
dataset = load_from_disk('tokenized_wiki')
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_loader = DataLoader(
    dataset['train'],
    batch_size=4,
    shuffle=True,
    collate_fn=collate_batch,
)

d_model   = 768                 
d_hidden  = d_model * 4         
n_blocks  = 8

# Model, loss, optimizer
model = MambaModel(d_model=d_model, d_hidden=d_hidden, n_blocks=n_blocks)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
epochs = 1

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    pbar = tqdm(train_loader, desc='Training')

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)

        # Shift for autoregressive LM
        inputs  = input_ids[:, :-1]  # (B, T-1)
        targets = input_ids[:, 1:]   # (B, T-1)

        logits = model(inputs)       # (B, T-1, vocab)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'mamba_model.pt')

