import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchcrf import CRF
from SegMapDataset import SegMapDataset
import sys
sys.path.append('/disk4/Lev/Projects/diffusion_finetuning')
from diffusers2.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from lora_diffusion import monkeypatch_lora, tune_lora_scale
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the CRF model
class CRFModel(nn.Module):
    def __init__(self, num_classes):
        super(CRFModel, self).__init__()
        self.crf = CRF(num_classes)
        
    def forward(self, x, tags):
        return self.crf(x, tags)

def replace_unet_in_pipeline(pipe, path_to_weights=None, device='cuda'):
    unet = pipe.unet
    tmp = unet.conv_out
    
    new_layer =  torch.nn.Conv2d(tmp.in_channels, tmp.out_channels * 2, kernel_size=tmp.kernel_size, padding=tmp.padding, bias=True, dtype=torch.float16) # Add 2 channels to serve for segmentation
    if path_to_weights:
        # load weights
        chkpt = torch.load(path_to_weights)
        new_layer.load_state_dict(chkpt['conv_out_state_dict'])
    else:
        # init weights to pipe weights
        new_layer.weight.data[:tmp.weight.data.shape[0], :, :, :] = tmp.weight.data[:, :, :, :]
    new_layer.to(device=device)
    unet.conv_out = new_layer
    return

def get_pipeline(weight_dir, load_text_enc=True):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    lora_weight_path = os.path.join(weight_dir, "lora_weight.pt")
    lora_text_weights_path = os.path.join(weight_dir, "lora_weight.text_encoder.pt")
    unet_weights_path = os.path.join(weight_dir, "unet_seg_weights.pt")
    
    monkeypatch_lora(pipe.unet, torch.load(lora_weight_path))
    monkeypatch_lora(pipe.text_encoder, torch.load(lora_text_weights_path), target_replace_module=["CLIPAttention"])

    tune_lora_scale(pipe.unet, 1)
    tune_lora_scale(pipe.text_encoder, 0.8)

    replace_unet_in_pipeline(pipe, unet_weights_path, device=pipe.device)
    
    return pipe

def get_segmap_from_path(path, seg_class, segmap_ds):
    seg_img = Image.open(path)
    bin_seg_img = segmap_ds.__get_png_segmap_by_class__(seg_img, seg_class)
    
    return bin_seg_img
    

def main(args):
    main_device = 'cuda:6'
    pipe_device = 'cuda:7'
    # Load  dataset
    train_dataset = SegMapDataset(
            instance_data_root=args.instance_data_dir,
            instance_segmap_data_root=args.instance_segmap_data_root,
            annotations_root = args.annotations_folder,
            instance_prompt='segmentation map of',
            size=args.resolution,
            center_crop=args.center_crop,
            resize=True,
            h_flip=False, # TODO make consisted flip with mask and image
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # define the stable diffusion pipeline
    pipe = get_pipeline(args.pretrained_pipeline_dir).to(pipe_device)
    segmap_img_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    pipe_output_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Define the CRF model and optimizer
    crf_model = CRFModel(num_classes=2).to(main_device)  # 2 classes: background and foreground
    crf_model.train()
    optimizer = optim.Adam(crf_model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train the CRF model
    for epoch in range(10):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Prepare the input and target
            in_img_path = batch['instance_images_path'][0]
            in_img = Image.open(in_img_path).convert('RGB').resize((args.resolution, args.resolution))
            
            seg_img_path = batch['instance_segmap_images_path'][0]
            seg_class = batch['instance_classes'][0]
            
            target = get_segmap_from_path(seg_img_path, seg_class, train_dataset)
            target = segmap_img_transforms(target)
            # plot target
            prompt = f'segmentation map of {seg_class}'
            
            # go thorugh the diffusion pipeline
            pipe_output = pipe(prompt=prompt, image=in_img, strength=0.1, guidance_scale=12, modified_unet=True, segmentation=True).images[0]
            # TODO? preprocess pipe_output
            pipe_output = pipe_output_transforms(pipe_output).to(main_device).unsqueeze(0)
            preds = F.softmax(pipe_output, dim=1)
            # Forward + backward + optimize
            outputs = crf_model(pipe_output, preds)
            # loss = crf_model.crf.neg_log_likelihood(outputs, target)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished training')
    
def plot_image(image):
    plt.imshow(image)
    plt.show()
    
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default='/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/original_train',
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_segmap_data_root",
        type=str,
        default='/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/seg_maps_class',
        required=False,
        help="A folder containing the training data of segmentation maps for instance images.",   
    )
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default="/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/Annotations",
        required=False,
        help="A folder containing annotations for segmentation maps",
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
        default=False,
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--pretrained_pipeline_dir",
        type=str,
        default="/disk4/Lev/Projects/diffusion_finetuning/output/grounding/grounding_no_text/mse_seg_pred__prev_noisy_segmap_latent_resumed_w_text_encoder_run2",
        required=False,
        help="A folder contating the weights for the pretrained stable diffusion pipeline",
    )
    parser.add_argument(
        "--load_text_encoder_weights",
        action="store_true",
        default=True,
        required=False,
        help="whether or not to load the lora text encoder weights",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate for the crf model.",
    )
    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
        
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)