#!/bin/bash

data_list=$1
gpu=$2
seed_list="1398"
task_list="all"

for seed in $seed_list; do
  for data in $data_list; do
    for task in $task_list; do
      python ./src/test.py \
        --test $data \
        --test_only \
        --local_rank $gpu \
        --load snap/naive/$data-small-$seed/$task \
        --backbone t5-small \
        --whole_word_embed \
        --max_text_length 512 \
        --batch_size 512 \
        --num_beams 20 \
        --output_path output_best_eval
    done
  done
done

for seed in $seed_list; do
  for data in $data_list; do
    for task in $task_list; do
      python ./src/test.py \
        --test $data \
        --test_only \
        --local_rank $gpu \
        --load snap/naive/$data-small-$seed/$task \
        --backbone t5-small \
        --whole_word_embed \
        --max_text_length 512 \
        --batch_size 512 \
        --num_beams 20 \
        --use_epoch_10 \
        --output_path output_epoch_10
    done
  done
done
