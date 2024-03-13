export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4,6 accelerate launch --main_process_port=24999 --multi_gpu --num_processes 2 --mixed_precision fp16 image2triplane.py