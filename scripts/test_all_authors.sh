#!/bin/bash

data_list=$1
gpu=$2
#seed_list="2022"
task_list="task-5 task-4 task-3 task-2 task-1 all"

#for seed in $seed_list; do
  for data in $data_list; do
#    for task in $task_list; do
      python ./src/test.py \
        --test $data \
        --test_only \
        --local_rank $gpu \
        --load snap/naive/"$1"-small-authors/all \
        --backbone t5-small \
        --whole_word_embed \
        --max_text_length 512 \
        --batch_size 512 \
        --num_beams 20 \
        --use_epoch_10 \
        --output_path output_epoch_10
#    done
  done
#done
