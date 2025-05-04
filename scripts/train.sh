#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_and_val.py \
    --data_mode "text" \
    --store_all_image_root_path "../data_example/text_mode/all_data" \
    --train_data_info "../data_example/text_mode/data_info.txt" \
    --val_data_info "../data_example/text_mode/data_info.txt" \
    --json_file "../data_example/text_mode/label2index.json" \
    --model_name "resnet34" \
    --num_workers 8 \
    --num_classes 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 5 \
    --learning_rate 0.001 \
    --scale_lr \
    --num_epochs 20 \
    --weight_decay 1e-5 \
    --save_model_path "./saved_model" \
    --mean 0.485, 0.456, 0.406 \
    --std 0.229, 0.224, 0.225
#    --data_mode "text" \
#    --store_all_image_root_path "../data_example/text_mode/all_data" \
#    --train_data_info "../data_example/text_mode/data_info.txt" \
#    --val_data_info "../data_example/text_mode/data_info.txt" \
#    --json_file "../data_example/text_mode/label2index.json" \
#
#    --data_mode "folder" \
#    --train_data_dir "../data_example/folder_mode" \
#    --val_data_dir "../data_example/folder_mode" \