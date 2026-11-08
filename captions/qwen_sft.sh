#!/bin/bash
# Qwen2-VL-2B SFT, hyperparameters from Section 4.2.
python -m captions.qwen_sft \
    --data_root "$1" \
    --dataset_name unset \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --output_dir logs/qwen2vl-fungi \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 \
    --save_strategy epoch \
    --logging_steps 5
