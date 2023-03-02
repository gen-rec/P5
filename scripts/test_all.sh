#!/bin/bash

data_list=$1
gpu=$2
seed_list="1398"
task_list="all task-1 task-2 task-3 task-4 task-5"

for seed in $seed_list; do
  for data in $data_list; do
    for task in $task_list; do
      python ./src/test.py \
        --test $data \
        --test_only \
        --local_rank $gpu \
        --load snap2/naive/$data-small-$seed/$task \
        --backbone t5-small \
        --whole_word_embed \
        --max_text_length 512 \
        --batch_size 1024 \
        --num_beams 20
    done
  done
done
