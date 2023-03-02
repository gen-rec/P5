#!/bin/bash
tasks="task-1 task-2 task-3 task-4 task-5"

for task in $tasks
do
  python ./src/test.py \
      --test "$1" \
      --test_only \
      --local_rank 0 \
      --load "$2"/"$task" \
      --backbone t5-small \
      --whole_word_embed \
      --max_text_length 512 \
      --batch_size 1024 \
      --num_beams 10
done