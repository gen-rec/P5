#!/bin/bash

# $1: test dataset
# $2: model path
# $3: backbone model
# $4: batch size
# $5: skip task
# $6: GPU id

python ./src/test.py \
    --test "$1" \
    --test_only \
    --local_rank "$6" \
    --load "$2" \
    --backbone "$3" \
    --whole_word_embed \
    --max_text_length 512 \
    --batch_size "$4" \
    --num_beams 20 \
    --skip_task "$5"