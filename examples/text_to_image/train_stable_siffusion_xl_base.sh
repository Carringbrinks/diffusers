export MODEL_NAME="/home/scb123/HuggingfaceWeight/stable-diffusion-xl-base-1.0"
export VAE_NAME="/home/scb123/HuggingfaceWeight/sdxl-vae-fp16-fix"
export DATASET_NAME="/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v2"

accelerate launch --config_file="/home/scb123/.cache/huggingface/accelerate/default_config.yaml" train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --proportion_empty_prompts=0. \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --num_train_epochs=100 \
  --learning_rate=1e-06 --lr_scheduler="cosine" --lr_warmup_steps=10 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=1200 \
  --validation_steps=100 \
  --validation_prompt="A 512x512 pixel image with a white background and a black rectangle. The top-left corner of the rectangle is positioned near coordinates (57.25, 0) pixels, and the bottom-right corner is near coordinates (454.75, 75) pixels." \
  --num_validation_images 5 \
  --output_dir="./weights/sdxl-base-model" \
  --enable_xformers_memory_efficient_attention \
