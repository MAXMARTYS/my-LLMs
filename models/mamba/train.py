from datasets import load_from_disk
from matplotlib.pyplot import step
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn

import os 
import tempfile
import json

from mamba import MambaModel
from utils import calculate_perplexity

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

def log_metrics(metrics_path, record: dict):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'a') as file:
        file.write(json.dumps(record) + '\n')

def train(epochs=1):
    # Load dataset
    dataset = load_from_disk('tokenized_wiki')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    val_batches = 500 
    batch_size = 16
    val_cutoff = val_batches * batch_size # 8000 samples in the val dataset

    full_train = dataset['train']
    val_subset = Subset(full_train, range(val_cutoff))
    train_subset = Subset(full_train, range(val_cutoff, len(full_train)))

    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,collate_fn=collate_batch)

    # Model, loss, optimizer
    model = MambaModel(
        d_model=512,
        d_hidden=2048,
        n_blocks=6
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True    
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    model.to(device)

    padding_value = 0
    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training loop
    start_epoch = 0 
    start_batch = 0
    total_batches_seen = 0

    checkpoint_path = 'models/mamba/training/checkpoint.pt'
    metrics_path = 'models/mamba/training/metrics.jsonl'
    save_every = 2000 # How often to save checkpoints

    if os.path.exists(checkpoint_path):
        print('Loading checkpoint...')
        start_epoch, start_batch, total_batches_seen = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f'Resuming from epoch {start_epoch}, batch {start_batch}')
    else:
        print('No checkpoints found. Starting from scratch.')

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        running_loss = 0.0
        running_tokens = 0

        pbar = tqdm(train_loader, desc='Training', leave=True)

        for batch_idx, batch in enumerate(pbar):

            if epoch == start_epoch and batch_idx < start_batch:
                continue  # Skip already trained batches

            input_ids = batch['input_ids'].to(device)

            inputs  = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()

            optimizer.zero_grad()

            logits = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_batches_seen += 1

            non_pad = (targets != padding_value).sum().item()
            running_loss += loss.item() * non_pad
            running_tokens += non_pad

            pbar.set_postfix(loss=loss.item())

            if total_batches_seen % save_every == 0:
                print(f'Saving checkpoint at epoch {epoch}, batch {batch_idx}...')
                save_checkpoint(checkpoint_path, model, optimizer, epoch, batch_idx, total_batches_seen)

                model.eval()

                avg_train_nll = running_loss / max(running_tokens, 1)
                train_ppl = torch.exp(torch.tensor(avg_train_nll)).item() # In torch: Perplexity = exp(CrossEntropyLoss)
                train_loss_avg = avg_train_nll

                val_ppl = calculate_perplexity(model, val_loader, device, max_batches=val_batches*batch_size)

                record = {
                    'step': total_batches_seen,
                    'epoch': epoch,
                    'train_loss': train_loss_avg,
                    'train_perplexity': train_ppl,
                    'val_perplexity': val_ppl
                }
                log_metrics(metrics_path, record)
                print(
                    f'[step {total_batches_seen}] '
                    f'train_loss={train_loss_avg:.4f}  '
                    f'train_ppl={train_ppl:.2f}  '
                    f'val_ppl={val_ppl:.2f}'
                )

                # Exit the eval state & reset metrics
                running_loss   = 0.0
                running_tokens = 0
                model.train()

    torch.save(model.state_dict(), 'transformer_model.pt')

if __name__=='__main__':
    train()

