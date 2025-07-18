accelerate launch train_control_flux.py\
    --pretrained_model_name_or_path="/home/scb123/huggingface_weight/FLUX.1-schnell" \
    --jsonl_for_train /home/scb123/PyProject/DeepData/fill50k_src/fill5k.jsonl \
    --dataloader_num_workers 8 \
    --image_column target \
    --caption_column prompt \
    --conditioning_image_column source \
    --output_dir /home/scb123/huggingface_weight/contronl_flux.1_schell \
    --mixed_precision fp16 \
    --resolution 512 \
    --learning_rate 1e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --num_train_epochs 1 \
    --validation_steps 1000 \
    --checkpointing_steps 1000 \
    --validation_image "/home/scb123/PyProject/DeepData/fill50k/source/0.png" "/home/scb123/PyProject/DeepData/fill50k/source/1.png" \
    --validation_prompt "pale golden rod circle with old lace background" "light coral circle with white background" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to="wandb" \
    --seed 42 \

    # --push_to_hub \

s