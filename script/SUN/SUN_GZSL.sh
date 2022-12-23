#!/bin/sh

for sc_loss in 0.001
do
    # 0.4
    for mask_pro in 0.4
    do
        # 0.5
        for attribute_miss in 0.5
        do
            # 1
            for mask_loss_xishu in 1
            do
                # 原 0.01
                for construct_loss_weight in 0.01
                    do
                    CUDA_VISIBLE_DEVICES=2 python ./model/main.py \
                    --dataset SUN \
                    --calibrated_stacking 0.4 \
                    --cuda --nepoch 30 --batch_size 8 --train_id 107 --manualSeed 2347 \
                    --pretrain_epoch 5  --pretrain_lr 1e-4 --classifier_lr 3e-5 \
                    --xe 1 --attri 1e-2 --regular 5e-5  --all \
                    --l_xe 1 --l_attri 1e-1  --l_regular 4e-2 \
                    --cpt 1e-9 --use_group --gzsl \
                    --patient 7 --loss_function two_loss \
                    --model_name Multi_attention_Model \
                    --mask_pro ${mask_pro} \
                    --mask_loss_xishu ${mask_loss_xishu} \
                    --xlayer_num 1 \
                    --construct_loss_weight ${construct_loss_weight} \
                    --mask_way newmask \
                    --sc_loss ${sc_loss} \
                    --attribute_miss ${attribute_miss} \
                    --gradient_time 7 \
                    --max_length 150 
                    done
            done
        done
    done
done