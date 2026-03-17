# Description

An ongoaing hobby project exploring post-transformer language model architectures.

# Models implemented

| Status                  | Architecture             |
|-------------------------|--------------------------|
| Implemented             | Transformer, Mamba (SSM) |
|-------------------------|--------------------------|
| In Implementation       | xLSTM                    |
|-------------------------|--------------------------|
| Planned Implementations | Hyena (H3)               |

# Data & training

The models were trained on wikipedia 2 dataset, tokenized with gpt-2 tokenizer. Tokenized version of the dataset can be found here:
https://huggingface.co/datasets/Maxmartys/tokenized-wiki 

The models were trained online via vast.ai. 

To run the training on vast.ai run following commands in jupyter terminal on pytorch instance:

```shell
git clone https://github.com/MAXMARTYS/my-LLMs.git

cd my-LLMs

pip3 install -r requirements.txt

python -c "
from datasets import load_dataset
ds = load_dataset('Maxmartys/tokenized-wiki')
ds['train'].save_to_disk('tokenized_wiki')
"

python3 models/{model_dir}/train.py
```

# Evaluation

TBA
