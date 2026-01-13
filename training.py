from datasets import load_from_disk
from matplotlib.pyplot import step
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn

import os 
import tempfile

from transformer import LLM

# Extend each sequence length to 512 (max length)
def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    attention_mask = (input_ids != 0).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

def save_checkpoint(path, model, optimizer, epoch, batch, total_batches_seen):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'total_batches_seen': total_batches_seen,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(checkpoint, tmp_path)

    os.replace(tmp_path, path)

def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    opt.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    batch = ckpt['batch']
    total_batches_seen = ckpt['total_batches_seen']
    return epoch, batch, total_batches_seen

# Load dataset
dataset = load_from_disk('tokenized_wiki')
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=False, collate_fn=collate_batch,)

# Model, loss, optimizer
model = LLM(depth=4, num_heads=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
epochs = 1
start_epoch = 0 
start_batch = 0
total_batches_seen = 0

checkpoint_path = 'transformer_checkpoints/checkpoint.pt'
save_every = 2000 # How often to save checkpoints

if os.path.exists(checkpoint_path):
    print('Loading checkpoint...')
    start_epoch, start_batch, total_batches_seen = load_checkpoint(checkpoint_path, model, optimizer, device)
    print(f'Resuming from epoch {start_epoch}, batch {start_batch}')
else:
    print('No checkpoints found. Starting from scratch.')

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    pbar = tqdm(train_loader, desc='Training', leave=True)

    for batch_idx, batch in enumerate(pbar):

        if epoch <= start_epoch and batch_idx < start_batch:
            continue  # Skip already trained batches

        input_ids = batch['input_ids'].to(device)

        inputs  = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        logits = model(inputs)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_batches_seen += 1

        pbar.set_postfix(loss=loss.item())

        if total_batches_seen % save_every == 0:
            print(f'Saving checkpoint at epoch {epoch}, batch {batch_idx}...')
            save_checkpoint(checkpoint_path, model, optimizer, epoch, batch_idx, total_batches_seen)

torch.save('transformer_model.pt')



