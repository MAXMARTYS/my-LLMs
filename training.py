from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn

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

train_loader = DataLoader(dataset['train'], batch_size=4, shuffle=True, collate_fn=collate_batch,)

# Model, loss, optimizer
model = LLM(depth=4, num_heads=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
epochs = 100

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    pbar = tqdm(train_loader, desc='Training', leave=True)

    for batch in pbar:
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

        pbar.set_postfix(loss=loss.item())

torch.save('my_llm_model.pt')
