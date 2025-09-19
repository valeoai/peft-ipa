# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
CUDA_VISIBLE_DEVICES=$7 python finetune.py \
    --base_model 'Qwen/Qwen2.5-7B' \
    --data_path 'commonsense_170k.json' \
    --output_dir $6 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate $4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 --adapter_name ipa \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --scaling $2 --weight_decay 0.0 --lora_dropout 0.0 --pre_data $3 --use_gradient_checkpointing --ipa_mode $5
