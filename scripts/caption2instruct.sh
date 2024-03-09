CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --main_process_port=24999 --multi_gpu --num_processes 3 --mixed_precision no caption2instruct.py


CUDA_VISIBLE_DEVICES=3 python caption2instruct1.py --num_chunks 4 --chunk_index 0
CUDA_VISIBLE_DEVICES=4 python caption2instruct1.py --num_chunks 4 --chunk_index 1
CUDA_VISIBLE_DEVICES=5 python caption2instruct1.py --num_chunks 4 --chunk_index 2
CUDA_VISIBLE_DEVICES=6 python caption2instruct1.py --num_chunks 4 --chunk_index 3