#!/bin/sh
for seed in 42 2021 2022 2023 2024 3131 666
do
    for gradient_time in 10
    do
        for sc_loss in 0.001
        do
            # 0.4
            for mask_pro in 0.4
            do
                # 0.5
                for attribute_miss in 0.5
                do
                    # 5
                    for mask_loss_xishu in 5
                    do
                        # 0.001
                        # bs = 7
                        for construct_loss_weight in 0.001
                            do
                            CUDA_VISIBLE_DEVICES=0 python ./model/main.py \
                            --dataset CUB \
                            --calibrated_stacking 0.7 \
                            --cuda --nepoch 30 --batch_size 7 --train_id 107 --manualSeed ${seed} \
                            --pretrain_epoch 5  --pretrain_lr 1e-4 --classifier_lr 3e-5 \
                            --xe 1 --attri 1e-2 --regular 5e-5 \
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
                            --gradient_time ${gradient_time} \
                            --max_length 250
                            done
                    done
                done
            done
        done
    done
done