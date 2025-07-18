accelerate launch train_controlnet_sd3_custom_data.py \
    --pretrained_model_name_or_path /home/scb123/huggingface_weight/stable-diffusion-3-medium-diffusers \
    --jsonl_for_train /home/scb123/PyProject/DeepData/fill50k_src/fill5k.jsonl \
    --image_column target \
    --caption_column prompt \
    --conditioning_image_column source \
    --output_dir="/home/scb123/huggingface_weight/contronlnet_sd3_medium_custom" \
    --mixed_precision="fp16" \
    --resolution 512 \
    --learning_rate 1e-5 \
    --lr_scheduler="cosine" \
    --num_train_epochs 1 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --report_to="wandb" \
    --seed 42 \
    --checkpointing_steps 1000 \
    --dataloader_num_workers 8 \
    # --push_to_hub \
