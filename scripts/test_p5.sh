#!/bin/bash
dataset="beauty"
cd ..
python ./src/test.py \
    --test $dataset \
    --test_only \
    --local_rank 0 \
    --load snap/atomic/beauty-small-2022/all \
    --backbone t5-small \
    --whole_word_embed \
    --max_text_length 512 \
    --batch_size 1024 \
    --num_beams 10