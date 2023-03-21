#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/Lev/Projects/diffusion_finetuning/data/original/"
export SEGMAP_INSTANCE_DIR="/home/Lev/Projects/diffusion_finetuning/data/seg_maps/"
export OUTPUT_DIR="/home/Lev/Projects/diffusion_finetuning/output/"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_segmap_data_root=$SEGMAP_INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_unet_segmentation \
  --instance_prompt="segmentation map" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000