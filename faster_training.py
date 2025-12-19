from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary

from transformer import LLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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



# Load dataset
dataset = load_from_disk('tokenized_wiki')
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
batch_size = 16
train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_batch,)

# Model, loss, optimizer
model = LLM(depth=6, num_heads=8)
# model.gradient_checkpointing_enable()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=100 * len(train_loader))

# Training loop
# I made math and 1 epoch is more than enough. We have ~2B tokens and the model is ~70M params.
# A lot of sources suggest 20 tokens per parameter, so 1.4B is enough.
epochs = 1 

token_param_ratio = 20
param_count = summary(
    model, 
    input_size=(2, 512), 
    dtypes=[torch.long], 
    verbose=0
).total_params
max_tokens = param_count * token_param_ratio

accumulation_steps = 4
tokens_seen = 0
end_training = False

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    pbar = tqdm(train_loader, desc='Training', leave=True)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):

        input_ids = batch['input_ids'].to(device)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] - 1

        inputs  = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        with autocast('cuda'):
            logits = model(inputs)
            logits = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(logits, targets_flat)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # scheduler.step()

        pbar.set_postfix(loss=loss.item())

        tokens_seen += batch_size * seq_len
        if tokens_seen > max_tokens:
            end_training = True
            break

    if end_training:
        break

torch.save('transformer_model.pt')

