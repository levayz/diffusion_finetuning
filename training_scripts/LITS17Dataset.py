import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset

import fnmatch
from torchvision import transforms

from PIL import Image

class LITS17Dataset(dataset):
    def __init__(self,
                 ct_dir,
                 seg_dir,
                 prompt,
                 tokenizer=None,
                 size=512,
                 center_crop=False,
                 num_slices=48,
                 resize=True,
                 ):
        
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_slices = num_slices
        self.resize = resize

        self.ct_list = self.__get_ct_file_names__(ct_dir)

        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        self.__length = len(self.ct_list)

        img_transforms = []
        segmap_transforms = []
        if self.resize:
            resize_t = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)
            img_transforms.append(resize_t)
            segmap_transforms.append(resize_t)

        self.image_transforms = transforms.Compose([*img_transforms, transforms.ToTensor()]) # TODO add transforms
        self.segmap_transforms = transforms.Compose([*segmap_transforms, transforms.PILToTensor()]) # TODO add transforms

    def __get_ct_file_names__(self, root_dir):
        ct_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, '*volume*'):
                ct_files.append(filename)
        return ct_files

    def __getitem__(self, index):
        example = {}

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array

        start_slice = random.randint(0, ct_array.shape[0] - self.num_slices)
        end_slice = start_slice + self.num_slices - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array)

        ct_imgs = self.__convert_slices_to_images__(ct_array)
        seg_imgs = self.__convert_slices_to_images__(seg_array)

        if self.tokenizer:
            example['instance_prompt_ids'] = self.tokenizer(
                self.prompt,
                padding='do_not_pad',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        example['instance_images'] = [self.image_transforms(ct_img) for ct_img in ct_imgs]
        example['instance_segmap_images'] = [self.segmap_transforms(seg_img) for seg_img in seg_imgs]

        return example

    def __len__(self):
        return self.__length
    
    def __convert_slices_to_images__(self, ct_arr):
        ct_imgs = []
        n_slices = len(ct_arr)
        arr = np.array(ct_arr, dtype=np.uint8)
        if len(arr.shape) > 3:
            arr = arr.squeeze(0)

        arr = np.moveaxis(arr, [0, 1, 2], [2, 0, 1])
        imgs = [Image.fromarray(arr[:,:,indx]) for indx in range(n_slices)]

        return imgs
