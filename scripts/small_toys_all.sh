#!/bin/bash

# Run with $ bash scripts/pretrain_P5_small_toys.sh 4
name=toys-small

output=snap/$name-$2

echo Output path: $output

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node="$1" \
    --master_port 12345 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed $2 \
        --train toys \
        --valid toys \
        --batch_size 32 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'rating,sequential,explanation,review,traditional' \
        --backbone 't5-small' \
        --output $output "${@:3}" \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log
