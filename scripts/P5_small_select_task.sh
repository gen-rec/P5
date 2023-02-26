#!/bin/bash

export OMP_NUM_THREADS=8

# Run with $ bash scripts/pretrain_P5_small_beauty.sh 4
dataset=$1
seeds="1398 42 2022"
num_nodes=$2
task_indices=(0 1 2 3 4)
losses=('rating' 'sequential' 'explanation' 'review' 'traditional')

for seed in $seeds; do
  for task_index in "${task_indices[@]}"; do
    task_display_name=$((task_index + 1))
    # name="${dataset}-small-${seed}-task-${task_display_name}"
    name="task-${task_display_name}"
    output="snap/atomic/${dataset}-small-${seed}/${name}"

    mkdir -p "$output"

    echo "Name: $output"

    PYTHONPATH=$PYTHONPATH:./src python -m torch.distributed.launch \
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
      --losses "${losses[$task_index]}" \
      --backbone 't5-small' \
      --output "$output" \
      --epoch 10 \
      --max_text_length 512 \
      --gen_max_length 64 \
      --whole_word_embed \
      --task_index "$task_index" \
      --run_valid | tee "${output}.log"
  done
done