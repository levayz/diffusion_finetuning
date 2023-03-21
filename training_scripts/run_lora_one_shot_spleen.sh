#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

accelerate launch train_lora_medical_segmentation.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--instance_data_dir="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train" \
--instance_segmap_data_root="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train" \
--output_dir="/disk4/Lev/Projects/diffusion_finetuning/output/miccai/no_text/" \
--train_unet_segmentation \
--run_name="mse_lora_rank4_bb_nat_imgs_finetuning_12_examples_spleen_run1" \
--instance_prompt="segmentation map of spleen" \
--organ="spleen" \
--resolution="512" \
--train_batch_size="1" \
--gradient_accumulation_steps="1" \
--learning_rate=1e-3 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/miccai2015/RawData/Training/seg_dict_12_examples.pkl" \
--vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/miccai2015/RawData/Training/vol_dict_12_examples.pkl" \
--resize="True" \
--max_train_steps=1000 \
--save_steps=1500 \
--lora_rank=4 \
--center_crop \
--color_jitter \
--resume_unet="/disk4/Lev/Projects/diffusion_finetuning/output/grounding/grounding_no_text/class_bin_segmaps_cos_loss_run1/lora_weight.pt"
# --resume_unet="/disk4/Lev/Projects/diffusion_finetuning/output/msd/no_text/mse_lora_rank_6_run1/lora_weight.pt" \
# --resume_unet="/disk4/Lev/Projects/diffusion_finetuning/output/lits/no_text/mse_lora_rank_6/lora_weight.pt"
# --seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/spleen_12_seg_dict.pkl" \
# --vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/spleen_12_vol_dict.pkl" \

                    