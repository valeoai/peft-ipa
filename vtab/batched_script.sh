#!/usr/bin/env bash

EXP_BASE_PATH=$1
DATA_PATH=$2
PEFT=$3

for DATASET in cifar100 caltech101 dtd 'oxford_flowers102' svhn sun397 oxford_iiit_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
    python train_script_siglip2.py --peft_model ${PEFT} --dataset ${DATASET} --exp_base_path ${EXP_BASE_PATH} --data_path ${DATA_PATH}
done

# bash batched_script_2.sh /home/yyin5/iveco/yyin5/peft/finetuned_result/vtab/test /home/yyin5/iveco/yyin5/peft/data/vtab-1k
# train_script_siglip2.py --peft_model lora --dataset caltech101 --exp_base_path /home/yyin5/iveco/yyin5/peft/finetuned_result/vtab/siglip2_fp_1e-4 --data_path /home/yyin5/iveco/yyin5/peft/data/vtab-1k