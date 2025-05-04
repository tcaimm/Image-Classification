#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 test_inference_multi.py \
    --data_mode "text" \
    --store_all_image_root_path "../data_example/text_mode/all_data" \
    --test_data_info "../data_example/text_mode/all_data/data_info.txt" \
    --json_file ".../data_example/text_mode/label2index.json" \
    --model_name "resnet34" \
    --model_path "./saved_model/your_model.pth" \
    --test_inference_dir "./test_info" \
    --num_workers 8 \
    --num_classes 3 \
    --batch_size 1 \
    --mean 0.485, 0.456, 0.406 \
    --std 0.229, 0.224, 0.225
#     --data_mode "text" \
#     --store_all_image_root_path "../data_example/text_mode/all_data" \
#     --test_data_info "../data_example/text_mode/all_data/data_info.txt" \
#     --json_file "../data_example/text_mode/label2index.json" \

#    --data_mode "folder" \
#    --test_data_dir "../simple_data" \