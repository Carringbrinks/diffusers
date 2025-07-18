export MODEL_NAME="/home/scb123/HuggingfaceWeight/FLUX.1-dev"
export OUTPUT_DIR="./trained-flux"
export DATA_NAME="/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v2"

accelerate launch  train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$DATA_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=10 \
  --max_train_steps=500 \
  --seed="0" \
