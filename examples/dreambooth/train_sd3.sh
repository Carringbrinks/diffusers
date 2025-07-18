export MODEL_NAME="/home/scb123/HuggingfaceWeight/stable-diffusion-3.5-medium"
export OUTPUT_DIR="./trained-sd3"
export DATA_NAME="/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v2"

accelerate launch  --config_file="/home/scb123/.cache/huggingface/accelerate/default_config_all.yaml" train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$DATA_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=512 \
  --use_8bit_adam \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="AdamW" \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=10 \
  --max_train_steps=500 \
  --seed="0" \
