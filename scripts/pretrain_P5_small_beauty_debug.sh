#!/bin/bash

# Run with $ bash scripts/pretrain_P5_small_beauty.sh 4
name=beauty-small

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python src/pretrain.py \
    --seed 2022 \
    --train beauty \
    --valid beauty \
    --batch_size 32 \
    --optim adamw \
    --warmup_ratio 0.05 \
    --lr 1e-3 \
    --num_workers 0 \
    --clip_grad_norm 1.0 \
    --losses 'rating,sequential,explanation,review,traditional' \
    --backbone 't5-small' \
    --output $output "${@:2}" \
    --epoch 10 \
    --max_text_length 512 \
    --gen_max_length 64 \
    --whole_word_embed > $name.log
