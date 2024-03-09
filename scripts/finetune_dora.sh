export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export WANDB_MODE=offline
# export NCCL_P2P_DISABLE="1"

python -m torch.distributed.run --master_port=24999 --nproc_per_node=6 \
         instruct_tri2tri/tinyllama_ft/train/finetune_mem.py \
        --lora_enable True --lora_r 8 --lora_alpha 32 --use_dora True \
        --data_path data/instrct_tri2tri/gpt-generated-prompts-450k.json \
        --model_name_or_path ckpts/TinyLlama-1.1B-Chat-v1.0 \
        --bf16 True \
        --output_dir checkpoints/tinyllama_caption2instruct_ft_dora \
        --max_steps 1000    \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 2000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --tf32 True  \
        --model_max_length 2048  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \
        --ddp_find_unused_parameters=False