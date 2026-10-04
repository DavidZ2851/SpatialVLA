#!/bin/bash
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.log
# Few-shot SpatialVLA LoRA fine-tune on one grogu GPU. Submit from the repo root, e.g.
#   sbatch --job-name=spatialvla_potato --export=ALL,ROOT=/scratch/haotian/lerobot/Kovavavvavava/pick_place_potato_onehouse_random,INTR=scripts/molmo/intrinsics/pick_place_potato_onehouse_random.json,RUN=potato_onehouse_random scripts/molmo/train_grogu.sbatch
# Defaults: batch 8, 20k steps (= 160k samples, as the pi0.5 LoRA baselines), a checkpoint every 2000.
cd ${SLURM_SUBMIT_DIR:-$HOME/SpatialVLA}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
. ${VENV:-$HOME/spatialvla_venv}/bin/activate
export NGPU=1 BS=${BS:-8} SAVE=${SAVE:-2000} OUT_ROOT=${OUT_ROOT:-$HOME/spatialvla_runs}
export MODEL=${MODEL:-$HOME/spatialvla_pretrained/spatialvla-4b-224-pt}
export EXTRA="--max_steps ${STEPS:-20000} ${EXTRA}" MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-12345} % 20000))
bash scripts/molmo/finetune_lora_molmo.sh
