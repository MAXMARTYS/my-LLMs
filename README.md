# Description

An ongoaing hobby project exploring post-transformer language model architectures.

# Models implemented

| Status                  | Architecture             |
|-------------------------|--------------------------|
| Implemented             | Transformer, Mamba (SSM) |
| In Implementation       | xLSTM                    |
| Planned Implementations | Hyena (H3)               |

If you want to read more about specific models, please check these papers:

| Models | Title |
|--------|-------|
| Transformer | Attention Is All You Need |
| Mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces |
| xLSTM | xLSTM: Extended Long Short-Term Memory |
| Hyena | Hyena Hierarchy: Towards Larger Convolutional Language Models |

# Data & training

The models were trained on wikipedia 2 dataset, tokenized with gpt-2 tokenizer. Tokenized version of the dataset can be found here:
https://huggingface.co/datasets/Maxmartys/tokenized-wiki 

The models were trained online via vast.ai. 

To run the training on vast.ai run following commands in jupyter terminal on pytorch instance:

Clone repository and set the directory.

```shell
git clone https://github.com/MAXMARTYS/my-LLMs.git
```
```shell
cd my-LLMs
```

Install uv and set up virtual environment.

```shell
pip3 install uv 
```
```shell
uv sync
```
```shell
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


# Evaluation

TBA
