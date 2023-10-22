#!/bin/bash

echo $CUDA_VISIBLE_DEVICES

model_name='densenet' #'convit_base'
batch_size= 64
num_epochs=1

echo batch_size = ${batch_size}
echo num_epochs = ${num_epochs}
echo model_name = ${model_name}

now=$(date +"%m-%d-%y-%H-%M")

python -m torch.distributed.launch ---use_env main.py \
					--model=${model_name}  --input-size 224 \
					--batch-size ${batch_size} --epochs ${num_epochs} \
					--output_dir ../models/herbarium_22/${model_name}/224/ce_${now} \
					--drop ${4:-.1}