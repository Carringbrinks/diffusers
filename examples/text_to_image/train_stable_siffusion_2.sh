export MODEL_NAME="/home/scb123/HuggingfaceWeight/stable-diffusion-2"
export TRAIN_DIR="/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v3"

accelerate launch --mixed_precision="fp16" --config_file="/home/scb123/.cache/huggingface/accelerate/default_config_all.yaml" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=768 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="./weights/stable-diffusion-2-full-v2-ds" \
  --checkpointing_steps 800 \
  --report_to=wandb