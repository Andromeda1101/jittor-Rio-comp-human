#!/bin/bash

echo "Start training skeleton model..."
python train_skeleton.py \
  --model_name pct2 \
  --optimizer adam \
  --learning_rate 0.0005 \
  --epochs 500 \
  --batch_size 32 \
  --output_dir output/skeleton_enhanced \
  --train_data_list data/train_list.txt \
  --val_data_list data/val_list.txt \
  --data_root data \
  --save_freq 20 \
  --val_freq 1 \
  --patience 20 \
  --print_freq 10

echo "Start training skin model..."
python train_skin.py \
  --model_name enhanced \
  --optimizer adam \
  --learning_rate 0.0005 \
  --epochs 800 \
  --batch_size 32 \
  --output_dir output/skin_enhanced \
  --train_data_list data/train_list.txt \
  --val_data_list data/val_list.txt \
  --data_root data \
  --save_freq 20 \
  --val_freq 1 \
  --patience 20 \
  --print_freq 10 \
  --export_render \
  --use_symmetry
