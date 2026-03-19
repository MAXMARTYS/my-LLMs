# Description

An ongoaing hobby project exploring post-transformer language model architectures.

# Models implemented

| Status                  | Architecture             |
|-------------------------|--------------------------|
| Implemented             | Transformer, KAT |
| In Implementation       | xLSTM, Mamba (SSM) |
| Planned Implementations | Hyena (H3), RetNet |

If you want to read more about specific models, please check these papers:

| Models | Title |
|--------|-------|
| Transformer | Attention Is All You Need |
| KAT | Kolmogorov-Arnold Transformer |
| Mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces |
| xLSTM | xLSTM: Extended Long Short-Term Memory |
| Hyena | Hyena Hierarchy: Towards Larger Convolutional Language Models |
| RetNet | Retentive Network: A Successor to Transformer for Large Language Models | 

# Data & training

The models were trained on wikipedia 2 dataset, tokenized with gpt-2 tokenizer. Tokenized version of the dataset can be found here:
https://huggingface.co/datasets/Maxmartys/tokenized-wiki 

The models were trained online via vast.ai. With NVIDIA RTX 3090 (24 GB, ~35 TFLOPs). New NVIDIA GPUs (like 5090) are incompatible with CUDA version used in this project (11.8). 

### To run the training on vast.ai run following commands in jupyter terminal on pytorch instance:

```shell
curl -s https://raw.githubusercontent.com/MAXMARTYS/my-LLMs/vastai_setup/setup.sh
```

### Or you can do that manually using the following commands one by one:

Clone repository and set the directory.

```shell
git clone https://github.com/MAXMARTYS/my-LLMs.git
cd my-LLMs
```

Install uv and set up virtual environment.

```shell
pip3 install uv 
uv sync

source .venv/bin/activate
```

Check cuda availability (optional).

```shell
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Download dataset from huggingface.

```shell
python -c "
from datasets import load_dataset
import sys
print('Downloading dataset...')
ds = load_dataset('Maxmartys/tokenized-wiki')
ds.save_to_disk('tokenized_wiki')
print(f'Done. {len(ds['train'])} samples saved.')
" || { echo 'Dataset download failed'; exit 1; }
```

Start training. Set MODEL_DIR to any model from the repo.

```shell
MODEL_DIR="transformer"
python3 models/$MODEL_DIR/train.py
```

# Download models

All trained models can be found on my huggingface account:
https://huggingface.co/Maxmartys 

The models will be named following a convention of '{MODEL_NAME}_{PARAM_COUNT}_wikipedia-2'.

# Evaluation

TBA
 