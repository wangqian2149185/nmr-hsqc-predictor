#!/usr/bin/env bash
# One-shot environment bootstrap (CUDA 12.1)
set -e

conda create -n nmr_pred python=3.10 -y
conda activate nmr_pred

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install -r requirements.txt

echo "Environment 'nmr_pred' ready."
