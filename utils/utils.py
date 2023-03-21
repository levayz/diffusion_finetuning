import torch
import sys
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append('/disk4/Lev/Projects/diffusion_finetuning')
from training_scripts.pretrained_segmentation_head import detection_mask_rcnn
import numpy as np
from typing import List

'''
convert an image outputted by the segmentation LDM to a segmentation mask
assumes that the background is darker than the segmented object
model_output: PIL.Image
returns: np.ndarray with pixels [0,0,0](bg) or [255,255,255](truth)
'''
def convert_seg_array_to_binary_rgb(model_output:Image.Image, up_threshold:int=0, down_threshold:int=0):
    # replace all pixels in seg_arr that dont have 0,0 in their GB channels with 255,255,255
    # replace very bright pixel or very dark pixels with 255,255,255
    seg_arr = np.asarray(model_output).copy()
    mask = (seg_arr[:,:,0] >= up_threshold) | (seg_arr[:,:,1] >= up_threshold) | (seg_arr[:,:,2] >= up_threshold) | \
            ((seg_arr[:,:,0] <= down_threshold) & (seg_arr[:,:,1] <= down_threshold) & (seg_arr[:,:,2] <= down_threshold))
    seg_arr[mask] = [255,255,255]
    # replace all pixels that arent [255,255,255] with [0,0,0]
    mask = (seg_arr != [255,255,255]).any(axis=2)
    new_seg_arr = seg_arr
    new_seg_arr[mask] = [0,0,0]
    
    return np.asarray(new_seg_arr)



'''
helper function to convert RGB image to binary mask
seg_rgb_arr - numpy array of shape (h, w, 3), where the segmented object is white and the background is black
'''
def convert_RGB_to_bin_seg_mask(seg_rgb_arr:np.ndarray):
    new_array = np.all(seg_rgb_arr == [255, 255, 255], axis=-1).astype(int)
    return np.asarray(new_array)

def convert_PIL_images_to_binary_masks(model_outputs: List[Image.Image], up_thresh=None, down_thresh=None):
    # convert model outputs to binary masks
    model_outputs_binary_masks = []
    for model_output in model_outputs:
        if up_thresh is not None and down_thresh is not None:
            rgb_segmask = convert_seg_array_to_binary_rgb(model_output, up_threshold=up_thresh, down_threshold=down_thresh)
        else:
            rgb_segmask = convert_seg_array_to_binary_rgb(model_output)
        bin_segmask = convert_RGB_to_bin_seg_mask(rgb_segmask)
        model_outputs_binary_masks.append(bin_segmask)
    return model_outputs_binary_masks

def replace_unet_in_pipeline(pipe, path_to_weights=None, device='cuda'):
    unet = pipe.unet
    tmp = unet.conv_out
    n_noise_pred_channels = unet.conv_out.out_channels
    new_layer = torch.nn.Conv2d(tmp.in_channels, tmp.out_channels * 2,
                                 kernel_size=tmp.kernel_size,
                                 padding=tmp.padding, bias=True,
                                 dtype=torch.float16) # Add 2 channels to serve for segmentation
    if path_to_weights:
        # load weights
        chkpt = torch.load(path_to_weights, map_location=device)
        new_layer.load_state_dict(chkpt['conv_out_state_dict'])
    else:
        # init weights to pipe weights
        new_layer.weight.data[:tmp.weight.data.shape[0], :, :, :] = tmp.weight.data[:, :, :, :]
    new_layer.to(device=device)
    unet.conv_out = new_layer
    return

import xml.etree.ElementTree as ET
import os

def get_annotations(img_paths, annotations_root):
    voc_classes = {'background' : 'background',
                    'aeroplane' : 'aeroplane',
                    'bicycle' : 'bicycle',
                    'bird' : 'bird',
                    'boat' : 'boat',
                    'bottle' : 'bottle',
                    'bus' : 'bus',
                    'car' : 'car',
                    'cat' : 'cat',
                    'chair' : 'chair',
                    'cow' : 'cow',
                    'diningtable' : 'dining table',
                    'dog' : 'dog',
                    'horse' : 'horse',
                    'motorbike' : 'motorbike',
                    'person' : 'person',
                    'pottedplant' : 'potted plant',
                    'sheep' : 'sheep',
                    'sofa' : 'sofa',
                    'train' : 'train',
                    'tvmonitor' : 'tv monitor'}

    images_names = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]
    xmls_paths = [os.path.join(annotations_root, img_name +'.xml') for img_name in images_names]
    
    annotations = []
    for path in xmls_paths:
        tree = ET.parse(path)
        root = tree.getroot()
        annontation = root.find('object/name')

        annotations += [voc_classes[annontation.text]]

    return annotations

# Sample random pics and get their seg
import os
from os import walk
import random

def diffuse_random_voc_imgs(img_folder, pipe, prompt, strength, guidance,
                        neg_prompt=None, n_examples=3, save_path=None, rcnn=False):
    
    f = next(walk(img_folder), (None, None, []))[2]
    filenames = [os.path.join(img_folder, filename) for filename in f]
    filenames = random.sample(filenames, n_examples)
    annotations_root = '/disk4/Lev/Projects/diffusion_finetuning/data/voc_pascal/Annotations/'
    annotations = get_annotations(filenames, annotations_root)

    fig , ax = plt.subplots(n_examples, 3, figsize=(48,48))
    for i, item in enumerate(zip(filenames, annotations)):
        f, annotation = item
        init_image = Image.open(f).convert("RGB").resize((512, 512))
        ax[i, 0].imshow(init_image, aspect='auto')
        ax[i, 0].axis('off')
        
        full_prompt = prompt + ' ' + annotation
        image = pipe(prompt=full_prompt, image=init_image, strength=strength, guidance_scale=guidance, modified_unet=True, segmentation=True).images[0]
        ax[i, 1].imshow(image, aspect='auto')
        ax[i, 1].axis('off')
        ax[i, 1].set_title(full_prompt + ' ' + f, fontsize=40)
        
        seg_arr = convert_seg_array_to_binary_rgb(image, 200)
        seg_img = Image.fromarray(seg_arr).convert('RGB')
        ax[i, 2].imshow(seg_img, aspect='auto')
        ax[i, 2].axis('off')
        if rcnn:
            detection_mask_rcnn(image)

        if save_path:
            if not os.path.exists(save_path):       
                os.mkdir(save_path)
            init_image.save(os.path.join(save_path, 'original_' + os.path.basename(f)))
            image.save(os.path.join(save_path, os.path.basename(f)))

    plt.show()
    

def get_good_slices(arr, only_w_tumors=False, s_indx=0):
    slices = []
    n_classes = 2 if only_w_tumors else 1
    for i in range(s_indx, len(arr)):
        if arr[i].max() >= n_classes:
            slices.append(i)
    return slices

'''
get array of segmnetation slices
return list of slices with class seg_class
'''
def get_good_slices_by_class(arr, seg_class=1):
    slices = []
    for i in range(len(arr)):
        if seg_class in arr[i]:
            slices.append(i)
    return slices

import SimpleITK as sitk

def diffuse_random_medical_images(imgs_path, seg_paths, pipe, prompt, strength, guidance, n_examples=5):
    img = sitk.ReadImage(imgs_path, sitk.sitkUInt8)
    seg_img = sitk.ReadImage(seg_paths, sitk.sitkUInt8)

    img = sitk.GetArrayFromImage(img)
    seg_img = sitk.GetArrayFromImage(seg_img)

    slices = get_good_slices(seg_img)
    print(f'#good slices:{len(slices)}')

    n_examples = min(n_examples, len(slices))
    indices = random.sample(slices, n_examples)
    fig, ax = plt.subplots(n_examples, 3, figsize=(13,13))
    seg_preds = []
    seg_imgs = []
    for i, indx in enumerate(indices):
        ax[i, 0].imshow(img[indx], cmap='gray')
        ax[i, 1].imshow(seg_img[indx], cmap='gray')
        seg_imgs += [seg_img[indx]]
        in_img = Image.fromarray(img[indx]).convert('RGB')
        output = pipe(prompt, in_img, strength=strength, guidance_scale=guidance, modified_unet=True, segmentation=True).images
        seg_preds += output
        ax[i, 2].imshow(output[0])

    plt.show()
    
from diffusers2.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from lora_diffusion import monkeypatch_lora, tune_lora_scale, monkeypatch_lora_lora, tune_lora_lora_scale

def load_pipe_from_path(path,
                        unet_seg_weights_name='unet_seg_weights.pt',
                        unet_lora_weights_name='lora_weight.pt',
                        clip_lora_weights_name='lora_weight.text_encoder.pt',
                        load_clip_weights = False,
                        clip_weights_path = '/disk4/Lev/Projects/diffusion_finetuning/training_scripts/clip_text_encoder_weights.pt',
                        load_lora_text=False,
                        unet_lora_scale=1.0,
                        text_encoder_scale=1.0,
                        path_to_finetuned_lora_lora_weights=None,
                        original_lora_rank=4,
                        rank_of_finetuned_loras=3,
                        scale_of_finetuned_loras=1.0,
                        device='cuda:0',):
    
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    if path is None:
        return pipe
    
    unet_weights_path = os.path.join(path, unet_seg_weights_name)
    unet_lora_weights_path = os.path.join(path, unet_lora_weights_name)
    clip_lora_weights_path = os.path.join(path, clip_lora_weights_name)

    # load clip weights
    if load_clip_weights:
        pipe.text_encoder.load_state_dict(
            torch.load(clip_weights_path, map_location=pipe.device))
    
    # load lora unet
    monkeypatch_lora(pipe.unet, torch.load(unet_lora_weights_path, map_location=pipe.device), r=original_lora_rank)
    tune_lora_scale(pipe.unet, unet_lora_scale)
    
    if load_lora_text:    
        monkeypatch_lora(pipe.text_encoder, torch.load(clip_lora_weights_path, map_location=pipe.device), target_replace_module=["CLIPAttention"])
        tune_lora_scale(pipe.text_encoder, text_encoder_scale)

    replace_unet_in_pipeline(pipe, unet_weights_path, device=pipe.device)
    
    # add lora lora
    if path_to_finetuned_lora_lora_weights is not None:
        loras = torch.load(path_to_finetuned_lora_lora_weights)
        monkeypatch_lora_lora(pipe.unet, loras, r=rank_of_finetuned_loras)
        tune_lora_lora_scale(pipe.unet, scale_of_finetuned_loras)
    return pipe


'''
input: path to folder with ct slices and their segmentations
output: dict with key as path to ct and value as list of good slices
'''
def get_good_slices_from_path(seg_folder, vol_folder, dataset):
    assert dataset in ['lits', 'msd', 'miccai']
    # get list of all files in folder
    f = next(walk(seg_folder), (None, None, []))[2]
    # take only files with 'segmentation' in them
    if vol_folder is None:
        filenames = [os.path.join(seg_folder, filename) for filename in f if 'segmentation' in filename]
    else:
        filenames = [os.path.join(seg_folder, filename) for filename in f]
        
    good_seg_paths = {} # dict key is path and value is list of good slices
    good_vol_paths = {}
    for i, item in enumerate(filenames):
        f = item
        # check if f has '.nii' in it
        if '.nii' not in f or os.path.basename(f).startswith('.'):
            continue
        img = sitk.ReadImage(f, sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        slices = get_good_slices_by_class(img)
        if len(slices) > 0:
            good_seg_paths[f] = slices
            if dataset == 'lits':
                vol_path = f.replace('segmentation', 'volume')
            elif dataset == 'msd':
                vol_path = os.path.join(vol_folder, os.path.basename(f))
            elif dataset == 'miccai':
                vol_path = os.path.join(vol_folder, os.path.basename(f).replace('label', 'img'))
            good_vol_paths[vol_path] = slices
        # replace string 'segmnetation' with 'volume'
        
    return good_seg_paths, good_vol_paths

import pickle

def create_pkl_w_good_slices_single_patient(imgs_path, seg_paths, pkl_path, dataset, n_examples=5):
    good_seg_paths, good_vol_paths = get_good_slices_from_path(seg_paths, vol_folder=imgs_path, dataset=dataset)
    
    seg_dict = {}
    vol_dict = {}
    n = 0
    r_seg_paths = list(zip(good_seg_paths.keys(), good_vol_paths.keys()))
    random.shuffle(r_seg_paths)
    for seg_path, vol_path in r_seg_paths:
        # seg_path = random.choice(list(good_seg_paths.keys()))
        patient_name = os.path.basename(seg_path).split('.')[0]
        print(patient_name)
        # vol_path = os.path.join(imgs_path, os.path.basename(seg_path))
        
        slices = good_seg_paths[seg_path]
        indices = random.sample(slices, min(n_examples, len(slices)))
        
        seg_dict[seg_path] = indices
        vol_dict[vol_path] = indices
        
        n += len(indices)
        if n >= n_examples:
            break
        
    print(seg_dict, "\n", vol_dict)
    with open(os.path.join(pkl_path, f'seg_dict_{n_examples}_examples.pkl'), 'wb') as f:
        pickle.dump(seg_dict, f)
        
    with open(os.path.join(pkl_path, f'vol_dict_{n_examples}_examples.pkl'), 'wb') as f:
        pickle.dump(vol_dict, f)

    