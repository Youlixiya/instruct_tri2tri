# CUDA_VISIBLE_DEVICES=4,5,6,7  accelerate launch --main_process_port=24999 --multi_gpu --num_processes 4 --mixed_precision fp16 instruct_tri2tri_train.py

CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port=24999 --num_processes 1 --mixed_precision fp16 instruct_tri2tri_train.py
