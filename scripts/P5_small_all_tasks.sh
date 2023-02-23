#!/bin/bash

export OMP_NUM_THREADS=8

# Run with $ bash scripts/pretrain_P5_small_beauty.sh 4
dataset=$1
seeds="1398 42 2022"
num_nodes=$2

for seed in $seeds; do
  name=$dataset-small-$seed
  output=snap/$name/all

  mkdir -p "$output"

  PYTHONPATH=$PYTHONPATH:./src \
    python -m torch.distributed.launch \
    --nproc_per_node="$num_nodes" \
    --master_port 12345 \
    src/pretrain.py \
    --distributed --multiGPU \
    --seed "$seed" \
    --train "$dataset" \
    --valid "$dataset" \
    --batch_size 32 \
    --optim adamw \
    --warmup_ratio 0.05 \
    --lr 1e-3 \
    --num_workers 8 \
    --clip_grad_norm 1.0 \
    --losses 'rating,sequential,explanation,review,traditional' \
    --backbone 't5-small' \
    --output "$output" \
    --epoch 10 \
    --max_text_length 512 \
    --gen_max_length 64 \
    --whole_word_embed \
    --run_valid | tee "${output}.log"
done
