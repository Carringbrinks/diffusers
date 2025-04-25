export MODEL_NAME="/home/scb123/HuggingfaceWeight/stable-diffusion-v1-4"
export TRAIN_DIR="/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v3"

accelerate launch --mixed_precision="fp16" --config_file="/home/scb123/.cache/huggingface/accelerate/default_config.yaml" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=500 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="./weights/stable-diffusion-v1-4-v3" \
  --checkpointing_steps 200 \
  --report_to=wandb