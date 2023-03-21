#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

accelerate launch train_lora_medical_segmentation.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--instance_data_dir="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train" \
--instance_segmap_data_root="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train" \
--output_dir="/disk4/Lev/Projects/diffusion_finetuning/output/lits/no_text" \
--train_unet_segmentation \
--run_name="mse_lora_rank_6_data_aug_run1" \
--instance_prompt="segmentation map of liver" \
--organ="spleen" \
--resolution="512" \
--train_batch_size="1" \
--gradient_accumulation_steps="1" \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--resize="True" \
--max_train_steps=10000 \
--num_train_epochs=1 \
--lora_rank=6 \
--center_crop \
--color_jitter \
--seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train/lits_train_seg_files_good_slices.pickle" \
--vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/lits17/train/lits_train_vol_files_good_slices.pickle" \
--description="train model for spleen segmentation on entire train spleen dataset" \
# --resume_unet="/disk4/Lev/Projects/diffusion_finetuning/output/lits/no_text/mse_seg_pred__prev_noisy_segmapt_latent_new_clip_run3/lora_weight.pt"
# --seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/spleen_12_seg_dict.pkl" \
# --vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/spleen_12_vol_dict.pkl" \
# --seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/seg_dict_100_examples.pkl" \
# --vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesFinetune/vol_dict_100_examples.pkl" \
#--instance_data_dir="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesTr" \
# --instance_segmap_data_root="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/labelsTr" \
# --seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/labelsTr_slices.pkl" \
# --vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task09_Spleen/imagesTr_slices.pkl" \
