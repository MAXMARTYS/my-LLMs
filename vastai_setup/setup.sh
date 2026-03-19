#!/bin/bash
set -e

echo "=== Cloning repo ==="
git clone https://github.com/MAXMARTYS/my-LLMs.git
cd my-LLMs

echo "=== Installing uv ==="
pip3 install uv

echo "=== Syncing virtual environment ==="
uv sync

echo "=== Activating venv ==="
source .venv/bin/activate

echo "=== Checking CUDA ==="
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== Downloading dataset ==="
python -c "
from datasets import load_dataset
ds = load_dataset('Maxmartys/tokenized-wiki')
print('Dataset loaded:', ds)
"

echo "=== All done! Ready to train. ==="
echo "=== To start training run: 'python3 models/MODEL_DIR/train.py' ==="