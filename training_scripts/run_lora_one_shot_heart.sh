#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

accelerate launch train_lora_medical_segmentation.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--instance_data_dir="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task06_Lung/imagesTr" \
--instance_segmap_data_root="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task06_Lung/labelsTr" \
--output_dir="/disk4/Lev/Projects/diffusion_finetuning/output/lits/no_text/" \
--train_unet_segmentation \
--run_name="oneshot_heart_run1" \
--instance_prompt="segmentation map of heart" \
--organ="heart" \
--resolution="512" \
--train_batch_size="1" \
--gradient_accumulation_steps="1" \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--seg_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task02_Heart/imagesFinetune/heart_seg_good_slice.pickle" \
--vol_slices_pkl="/disk4/Lev/Projects/diffusion_finetuning/data/MSD/Task02_Heart/imagesFinetune/heart_vol_good_slice.pickle" \
--resize="True" \
--max_train_steps=500 \
--resume_unet="/disk4/Lev/Projects/diffusion_finetuning/output/lits/no_text/mse_seg_pred__prev_noisy_segmapt_latent_new_clip_run3_resumed/lora_weight_e0_s500.pt"

                    