#!/bin/bash
# LoRA fine-tune of SpatialVLA-4B on a MolmoSpaces LeRobot dataset (data/molmo_dataset.py), following
# scripts/spatialvla_4b_finetune/finetune_lora.sh (lr 5e-4, r=32, alpha=32, linear targets, chunk 4).
#   ROOT=<lerobot dataset> INTR=<intrinsics json> RUN=<name> [NGPU=1 BS=8 EPOCHS=50 SAVE=1000 MODEL=...] bash scripts/molmo/finetune_lora_molmo.sh
set -ex
cd "$(dirname "$0")/../.."
NGPU=${NGPU:-1}; BS=${BS:-8}; EPOCHS=${EPOCHS:-50}; SAVE=${SAVE:-1000}
MODEL=${MODEL:-/data/haotian/spatialvla_pretrained/spatialvla-4b-224-pt}
OUT=${OUT_ROOT:-outputs/molmo}/$RUN
mkdir -p $OUT
export PYTHONPATH="${PYTHONPATH}:$(pwd)" LAUNCHER=pytorch TOKENIZERS_PARALLELISM=true
torchrun --standalone --nnodes=1 --nproc-per-node $NGPU --master_port ${MASTER_PORT:-29531} \
  train/spatialvla_finetune.py \
  --model_name_or_path $MODEL \
  --lora 32 --lora_alpha 32 --lora_target linear \
  --ignore_data_skip True \
  --molmo_root $ROOT --molmo_intrinsics $INTR --molmo_name ${NAME:-molmo_franka/1.0.0} \
  --obs_backward_steps 0 --obs_backward_delta 1 --action_forward_steps 3 \
  --flash_attn ${FLASH:-False} \
  --output_dir $OUT --overwrite_output_dir False \
  --freeze_vision_tower False \
  --dataloader_num_workers ${WORKERS:-4} \
  --bf16 True --tf32 True \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BS --gradient_accumulation_steps 1 \
  --save_strategy steps --save_steps $SAVE --save_total_limit ${KEEP:-20} \
  --learning_rate ${LR:-5e-4} --weight_decay 0.0 --warmup_ratio 0.005 --lr_scheduler_type linear \
  --logging_steps ${LOG_STEPS:-50} --do_train True \
  --grad_checkpoint True --deepspeed ${DS_CONFIG:-scripts/molmo/zero1_torch_adam.json} \
  --report_to ${REPORT:-tensorboard} --log_level warning ${EXTRA}
