import os
import sys
import torch
from models.transformer.transformer import Transformer 
# from models.kat.kat import KAT 
from models.mamba.mamba import MambaModel 
from transformers import AutoTokenizer
import os

if __name__=='__main__':

    def load_checkpoint(path, model, opt, device):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        batch = ckpt['batch']
        total_batches_seen = ckpt['total_batches_seen']
        return epoch, batch, total_batches_seen


    CHECKPOINT = 'models/mamba/training/checkpoint.pt'
    PROMPT = 'The capital of France is'
    # PROMPT = 'Wikipedia was invented'
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 1.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('gpt2')  

    model = MambaModel(d_model=512, d_hidden=2048, n_blocks=6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters())

    epoch, batch, total_batches = load_checkpoint(CHECKPOINT, model, opt, DEVICE)
    print(f'Loaded checkpoint: epoch={epoch}, batch={batch}, total_batches={total_batches}')


    input_ids = tokenizer.encode(PROMPT, return_tensors='pt').to(DEVICE)

    output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print('\n--- Generated Text ---')
    print(generated_text)