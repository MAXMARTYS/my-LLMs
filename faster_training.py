from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformer import LLM

def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]
    input_ids = pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=0 
    )

    attention_mask = (input_ids != 0).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

# Load dataset
dataset = load_from_disk('tokenized_wiki')
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_loader = DataLoader(
    dataset['train'], 
    batch_size=4,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# Model, loss, optimizer
model = LLM(depth=4, num_heads=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.compile(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True if device.type == 'cuda' else False
)
scheduler = CosineAnnealingLR(optimizer, T_max=100 * len(train_loader))
scaler = GradScaler()

# Training loop
accumulation_steps = 4
epochs = 100

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    model.train()

    pbar = tqdm(train_loader, desc='Training', leave=True)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)

        inputs  = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        with autocast():
            logits = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(logits, targets_flat)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        pbar.set_postfix(
            loss=loss.item() * accumulation_steps,
            lr=scheduler.get_last_lr()[0]
        )

    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f'checkpoint_epoch_{epoch+1}.pt')

torch.save(model.state_dict(), 'my_llm_model.pt')