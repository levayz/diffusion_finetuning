#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/original_train"
export OUTPUT_DIR="/disk4/Lev/Projects/diffusion_finetuning/output/lits17/w_text"

accelerate launch train_lora_segmentation.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir="/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/original_train" \
  --instance_segmap_data_root="/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/seg_maps_class" \
  --output_dir=$OUTPUT_DIR \
  --run_name="ct_liver_segmentation_good_slices_only_run2" \
  --instance_prompt="segmnetation map of liver" \
  --train_unet_segmentation \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
#   --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000 \ 
  --description="same trainig as run1, but better segmentation map differece between pixels"