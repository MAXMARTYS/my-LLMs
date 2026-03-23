import mauve
from datasets import load_dataset
import torch
from models.transformer.transformer import Transformer 
from models.kat.kat import KAT 
from transformers import AutoTokenizer
import os
from tqdm import tqdm

if __name__=='__main__':

    def load_checkpoint(path, model, opt, device):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        batch = ckpt['batch']
        total_batches_seen = ckpt['total_batches_seen']
        return epoch, batch, total_batches_seen


    CHECKPOINT = 'models/kat/training/checkpoint.pt'
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 1.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('gpt2')  

    model = KAT(depth=6, num_heads=8, m=4, n=3, groups=8).to(DEVICE)
    # model = Transformer(depth=6, num_heads=8).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters())

    epoch, batch, total_batches = load_checkpoint(CHECKPOINT, model, opt, DEVICE)
    print(f'Loaded checkpoint: epoch={epoch}, batch={batch}, total_batches={total_batches}')

    wiki = load_dataset('Maxmartys/tokenized-wiki', split='train')

    N = 200
    reference_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in wiki['input_ids'][:N]
    ]

    print('Generating samples')
    generated_texts = []
    for i in tqdm(range(N), desc='Generating'):
        seed_ids = wiki['input_ids'][i][:10]  # first 10 tokens as prompt
        input_ids = torch.tensor([seed_ids]).to(DEVICE)
        output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(text)

    print('Calculating score')
    result = mauve.compute_mauve(
        p_text=reference_texts,   # real human text
        q_text=generated_texts,   # model generated text
        device_id=0,              # GPU id, or remove this line for CPU
        max_text_length=256,
        verbose=False
    )

    print(f'MAUVE score: {result.mauve:.4f}') 