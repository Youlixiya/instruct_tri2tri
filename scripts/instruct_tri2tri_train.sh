CUDA_VISIBLE_DEVICES=0,2,4  accelerate launch --main_process_port=24999 --multi_gpu --num_processes 3 --mixed_precision fp16 instruct_tri2tri_train.py
