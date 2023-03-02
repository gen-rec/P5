#!/bin/bash
python ./src/test.py \
    --test "$1" \
    --test_only \
    --local_rank 0 \
    --load "$2" \
    --backbone t5-small \
    --whole_word_embed \
    --max_text_length 512 \
    --batch_size "$3" \
    --num_beams 20 \
    --skip_task "$4"