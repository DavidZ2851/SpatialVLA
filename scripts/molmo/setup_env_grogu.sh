#!/bin/bash
# SpatialVLA env for grogu's RTX PRO 6000 Blackwell GPUs (sm_120 -> CUDA 12.8 torch). Our MolmoSpaces data
# path (data/molmo_dataset.py) needs no TensorFlow/dlimp; flash-attn is not installed (--flash_attn False).
#   bash scripts/molmo/setup_env_grogu.sh [venv dir, default ~/spatialvla_venv]
set -ex
V=${1:-$HOME/spatialvla_venv}
uv venv --python 3.10 $V
. $V/bin/activate
uv pip install "torch==2.7.1" "torchvision==0.22.1" "transformers==4.47.0" "tokenizers==0.21.0" "peft==0.14.0" \
    "accelerate==1.0.1" "deepspeed==0.15.3" "numpy==1.26.4" einops "scipy==1.14.1" "pillow==11.0.0" pandas pyarrow av \
    "datasets==3.0.2" tensorboard h5py huggingface_hub tqdm \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
python -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
# HF_HOME may point at node-local /scratch (absent on the login node): keep the hub cache in $HOME
HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_XET=1 huggingface-cli download IPEC-COMMUNITY/spatialvla-4b-224-pt --local-dir ${MODEL_DIR:-$HOME/spatialvla_pretrained/spatialvla-4b-224-pt}
