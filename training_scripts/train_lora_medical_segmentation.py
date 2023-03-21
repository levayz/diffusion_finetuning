# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import sys
sys.path.append('/disk4/Lev/Projects/diffusion_finetuning/') # Need this to import files from the diffusers folder
print(sys.path)

# from my_diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
# from my_diffusers.models.unet_2d_condition import UNet2DConditionModel

from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_on_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from pathlib import Path

import random
import re

from torch.utils.tensorboard import SummaryWriter
import functools
import operator

import matplotlib.pyplot as plt
import numpy as np

import xml.etree.ElementTree as ET
import wandb

from data.get_voc_color_by_class import get_voc_colormap_png, color_map, get_color_map_by_class_rgb
from LITS17Dataset import LITS17Dataset
def save_img(img, path):
    img = img.cpu()
    np_img = img.detach().numpy().squeeze()
    np_img = np.transpose(np_img, (1,2,0))
    pil_img = Image.fromarray(np.uint8(np_img))
    pil_img.save(path)

def save_tensor_as_img(tensor, path, normalized=False):
    if tensor.device != torch.device("cpu"):
        tensor = tensor.detach().cpu()
    if normalized:
        arr = np.array(tensor*62, dtype=np.uint8)
    else:
        arr = np.array(tensor, dtype=np.uint8)
    
    arr = np.moveaxis(arr, [0, 1, 2], [2, 0, 1])
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    img = Image.fromarray(arr)
    img.save(path)

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_segmap_data_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of segmentation maps for instance images.",   
    )
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default="../data/Annotations/",
        required=False,
        help="A folder containing annotations for segmentation maps",
    )
    parser.add_argument(
        "--train_unet_segmentation",
        default=True,
        required=False,
        action="store_true",
        help="add extra channels to unet model for segmentation and train them",
    )
    parser.add_argument(
        "--finetune_unet_segmentation",
        default=False,
        required=False,
        action="store_true",
        help="add another lora layer to unet for finetuning",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        required=False,
        help="run name for TesnorBoard",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        required=False,
        help="description of current run"
    )
    parser.add_argument(
        "--seg_slices_pkl",
        type=str,
        default=None,
        required=False,
        help="path to pickle object contraining dictionary with paths and desired slices from ct segmentation maps"
    )
    parser.add_argument(
        "--vol_slices_pkl",
        type=str,
        default=None,
        required=False,
        help="path to pickle object contraining dictionary with paths and desired slices from ct volume"
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--organ",
        type=str,
        default=None,
        required=True,
        help="organ training on",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
        print(args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            logger.warning(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args


def main(args):
    main_device = 'cuda:0'
    unet_device = 'cuda:1'
    # text_enc_device = 'cuda:1'
    text_enc_device = main_device 
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        device_placement=False
    )
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder.load_state_dict(torch.load('/disk4/Lev/Projects/diffusion_finetuning/training_scripts/clip_text_encoder_weights.pt',
                                            map_location=torch.device('cpu')))
    print('***Loaded text encoder weights from /disk4/Lev/Projects/diffusion_finetuning/training_scripts/clip_text_encoder_weights.pt***')
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet.requires_grad_(False)
    unet_lora_params, _ = inject_trainable_lora(
        unet, r=args.lora_rank, loras=args.resume_unet
    )
    print(f'loaded first lora with rank {args.lora_rank}')
    
    # double the channels of the last layer of unet and set them to require_grad_(True)
    n_noise_pred_channels = unet.conv_out.out_channels
    if args.train_unet_segmentation:
        tmp = unet.conv_out
        new_layer =  torch.nn.Conv2d(tmp.in_channels, tmp.out_channels * 2, kernel_size=tmp.kernel_size, padding=tmp.padding, bias=True) # Add channels to serve for segmentation
        new_layer.weight.data[:tmp.weight.data.shape[0], :, :, :] = tmp.weight.data[:, :, :, :]
        
        unet.conv_out = new_layer
        # unet.conv_out.weight.data.requires_grad_(False)
        # unet.conv_out.weight.data[tmp.out_channels:, :, :, :].requires_grad_(True)
        unet.conv_out.requires_grad_(True)
        # Add a fc layer with softmax to the added channels
        n_features =  4 * 64 * 64
        segnet = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_features),
            #torch.nn.Softmax(dim=1)
        )
        segnet.requires_grad_(True)
        segnet_params = segnet.parameters()

    if args.resume_unet:
        # remove base name from args.resume_unet
        chkpt = torch.load(os.path.join(os.path.dirname(args.resume_unet),"unet_seg_weights.pt"), map_location=unet.device)
        unet.conv_out.load_state_dict(chkpt['conv_out_state_dict'])
        unet.conv_out.requires_grad_(True)
        print("resuming unet from checkpoint with grad on conv_out")
        print('***unet loaded from checkpoint... resuming...***')

    if args.finetune_unet_segmentation:
        # TODO add another layer of lora, dont optimize the previous lora
        unet.requires_grad_(False)
        unet_lora_params, _ = inject_trainable_lora_on_lora(
            unet, r=3,
        )

    
    for _up, _down in extract_lora_ups_down(unet, extract_lora_lora=True if args.finetune_unet_segmentation else False):
        # print("Before training: Unet First Layer lora up", _up.weight.data)
        # print("Before training: Unet First Layer lora down", _down.weight.data)
        break

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )
        for _up, _down in extract_lora_ups_down(
            text_encoder, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: text encoder First Layer lora up", _up.weight.data)
            print(
                "Before training: text encoder First Layer lora down", _down.weight.data
            )
            break

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )

    params_to_optimize = (
        [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": text_lr,
            },
        ]
        if args.train_text_encoder
        else itertools.chain(*unet_lora_params)
    )

    if args.train_unet_segmentation:
        params_to_optimize = (
            [
                {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
                {
                    "params": itertools.chain(*text_encoder_lora_params),
                    "lr": text_lr,
                },
            ]
            if args.train_text_encoder
            else itertools.chain(unet.conv_out.parameters(), *unet_lora_params)
        )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = LITS17Dataset(
        ct_dir=args.instance_data_dir,
        seg_dir=args.instance_segmap_data_root,
        prompt=args.instance_prompt,
        organ=args.organ,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=True,
        h_flip=True,
        rand_rotate=False,
        path_slices_for_segmap=args.seg_slices_pkl,
        path_slices_for_vol=args.vol_slices_pkl,
    )
    
    print("train_dataset_size", len(train_dataset))

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        organ_input_ids = [example["organ_prompt_ids"] for example in examples]
        pixel_values = [example["instance_image"] for example in examples]
        img = pixel_values[0]
        save_tensor_as_img(img, './img.jpeg', normalized=True)
        if args.instance_segmap_data_root:
            seg_map_pixel_values = [example["instance_segmap_image"] for example in examples]
            save_tensor_as_img(seg_map_pixel_values[0], './segmap_img.jpeg', normalized=True)
        
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        if args.instance_segmap_data_root:
            seg_map_pixel_values = torch.stack(seg_map_pixel_values)
            seg_map_pixel_values = seg_map_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        organ_input_ids = tokenizer.pad(
            {"input_ids": organ_input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if args.instance_segmap_data_root:
            batch = {
                "input_ids": input_ids,
                "organ_input_ids": organ_input_ids,
                "pixel_values": pixel_values,
                "segmap_pixel_values" : seg_map_pixel_values,
            }
        else:
            batch = {
                "input_ids": input_ids,
                "organ_input_ids": organ_input_ids,
                "pixel_values": pixel_values,
            }

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    elif not args.train_unet_segmentation:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else: # TODO this could probably be deleted
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler,
        )
    # put all on correct device
    unet.to(unet_device)
    print(f'unet device:{unet.device}')
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    #vae.to(accelerator.device, dtype=weight_dtype)
    vae.to(main_device, dtype=weight_dtype)
    # segnet.to(main_device, dtype=weight_dtype)
    
    if not args.train_text_encoder:
        #text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(main_device, dtype=weight_dtype)
    else:
        text_encoder.to(text_enc_device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    writer = SummaryWriter(log_dir=args.output_dir)
    
    wandb_config = {
        'lr' : args.learning_rate,
        'training_steps:' : args.max_train_steps,
        'text_encoder' : True if args.train_text_encoder else False,
        'text_enc_lr' : text_lr if args.train_text_encoder else None,
        'segmaps:' : args.instance_segmap_data_root if args.instance_segmap_data_root else None,
        'description': args.description,
    }
    wandb.init(config=wandb_config, project='diffusion_segmentation', name=args.run_name) 

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0
    min_loss = None

    for epoch in range(args.num_train_epochs):
        if not args.train_unet_segmentation:
            unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(main_device, dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            segmap = batch["segmap_pixel_values"].to(main_device, dtype=weight_dtype)
            segmap_latents = vae.encode(
                segmap
                ).latent_dist.sample().to(main_device)
            segmap_latents = segmap_latents * 0.18215 # TODO why this number?

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"].to(text_enc_device))[0].to(main_device, dtype=weight_dtype)
            organ_hidden_states  = text_encoder(batch["organ_input_ids"].to(text_enc_device))[0].to(main_device, dtype=weight_dtype)
            # print(f'\n\nencoder hidden states shape: {encoder_hidden_states.shape}\n\n')
            # Predict the noise residual
            if not args.train_unet_segmentation:
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample # this is the noise prediction
            # Predict segmentation map
            else:
                noisy_latents = noisy_latents.to(unet_device)
                timesteps = timesteps.to(unet_device)
                encoder_hidden_states = encoder_hidden_states.to(unet_device)
                unet_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample.to(main_device)
                model_pred = unet_pred[:,:n_noise_pred_channels, :, :]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = (
                    F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                # Compute prior loss
                prior_loss = F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="mean"
                )

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                if args.train_unet_segmentation:
                    model_pred = unet_pred[:, :n_noise_pred_channels, :, :] # this is noise
                    seg_pred = unet_pred[:, n_noise_pred_channels:, :, :].to(main_device)

                    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(main_device)
                    noise_scheduler.betas = noise_scheduler.betas.to(main_device)
                    noise_scheduler.alphas = noise_scheduler.alphas.to(main_device)
                    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(main_device)
                    
                    timesteps = timesteps.to(main_device)
                    segmap_latents = segmap_latents.to(main_device)
                    noise.to(main_device)
                    noisy_segmap_latents = noise_scheduler.add_noise(segmap_latents, noise, timesteps)
                    noisy_latents = noisy_latents.to(main_device)
                    prev_noisy_segmap_latents = noise_scheduler.step(noise, timesteps, noisy_segmap_latents).prev_sample
                    
                    mse_loss = F.mse_loss(prev_noisy_segmap_latents.float(), seg_pred.float())
                    loss = mse_loss                    
                    # similarity_labels = torch.matmul(organ_hidden_states, latents.t())
                    # similarity_logits = torch.matmul(organ_hidden_states, torch.matmul(latents,prev_noisy_segmap_latents.t()).t())
                    # cosine_sim_loss = torch.nn.CosineEmbeddingLoss()
                    # labels = torch.ones(1).to(main_device)
                    # sim_loss = cosine_sim_loss(prev_noisy_segmap_latents.view(1,-1).float(), seg_pred.view(1,-1).float(), labels)
                    
                    # loss = (mse_loss + sim_loss) / 2
                    # loss = mse_loss.to(main_device)
                    wandb.log({'seg_loss' : loss})
                    # wandb.log({'mse_loss' : mse_loss})
                    # wandb.log({'sim_loss' : sim_loss})
                    # wandb.log({'reconstruction_loss' : reconstruction_loss})
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            optimizer.zero_grad()

            global_step += 1
            
            if global_step % 100 == 0:
                if args.train_unet_segmentation:
                    writer.add_scalar(tag='segmentation loss',
                        scalar_value=mse_loss,
                        global_step=global_step)

                else:
                    writer.add_scalar(tag='noise loss',
                        scalar_value=loss,
                        global_step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                        # it, the models will be unwrapped, and when they are then used for further training,
                        # we will crash. pass this, but only to newer versions of accelerate. fixes
                        # https://github.com/huggingface/diffusers/issues/1566
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True}
                            if accepts_keep_fp32_wrapper
                            else {}
                        )
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, **extra_args
                            ),
                            revision=args.revision,
                        )

                        filename_unet = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                        )
                        filename_text_encoder = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                        print(f"save weights {filename_unet}, {filename_text_encoder}")
                        save_lora_weight(pipeline.unet, filename_unet, extract_lora_lora=True if args.finetune_unet_segmentation else False)
                        if args.train_text_encoder:
                            save_lora_weight(
                                pipeline.text_encoder,
                                filename_text_encoder,
                                target_replace_module=["CLIPAttention"],
                            )
                            
                        if args.train_unet_segmentation:
                            torch.save({'conv_out_state_dict' : unet.conv_out.state_dict(),
                                        'segmentation_head' : segnet.state_dict()}
                                        ,os.path.join(args.output_dir, f"unet_seg_weights.pt"))


                        for _up, _down in extract_lora_ups_down(pipeline.unet, extract_lora_lora=True if args.finetune_unet_segmentation else False):
                            print(
                                "First Unet Layer's Up Weight is now : ",
                                _up.weight.data,
                            )
                            print(
                                "First Unet Layer's Down Weight is now : ",
                                _down.weight.data,
                            )
                            break
                        if args.train_text_encoder:
                            for _up, _down in extract_lora_ups_down(
                                pipeline.text_encoder,
                                target_replace_module=["CLIPAttention"],
                            ):
                                print(
                                    "First Text Encoder Layer's Up Weight is now : ",
                                    _up.weight.data,
                                )
                                print(
                                    "First Text Encoder Layer's Down Weight is now : ",
                                    _down.weight.data,
                                )
                                break

                        last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\nLora TRAINING DONE!\n\n")
        writer.close()

        if args.output_format == "pt" or args.output_format == "both":
            save_lora_weight(pipeline.unet, args.output_dir + "/lora_weight.pt", extract_lora_lora=True if args.finetune_unet_segmentation else False)
            if args.train_text_encoder:
                save_lora_weight(
                    pipeline.text_encoder,
                    args.output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )

        if args.output_format == "safe" or args.output_format == "both":
            loras = {}
            loras["unet"] = (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"})
            if args.train_text_encoder:
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            save_safeloras(loras, args.output_dir + "/lora_weight.safetensors")

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training",
                blocking=False,
                auto_lfs_prune=True,
            )

        if args.train_unet_segmentation:
            torch.save({'conv_out_state_dict' : unet.conv_out.state_dict(),
                        'segmentation_head' : segnet.state_dict()}
                        ,os.path.join(args.output_dir,"unet_seg_weights.pt"))
            torch.save({'unet_weights' : unet.state_dict()}, 
                        os.path.join(args.output_dir,"entire_finetuned_unet_weights.pt"))

    accelerator.end_training()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = parse_args()
    main(args)
   