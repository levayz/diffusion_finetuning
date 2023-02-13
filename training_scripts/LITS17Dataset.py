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
import itertools
import pickle

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
                 path_slices_for_segmap=None,
                 path_slices_for_vol=None,
                 ):
        
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_slices = num_slices
        self.resize = resize

        self.ct_dir = ct_dir
        self.seg_dir = seg_dir

        self.ct_list = self.__get_ct_file_names__(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))
        
        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        if path_slices_for_segmap:
            self.ct_list_w_slices = self.__create_list_w_slices_from_pickle__(path_slices_for_vol)
            self.seg_list_w_slices = self.__create_list_w_slices_from_pickle__(path_slices_for_segmap)
        else:
            self.ct_list_w_slices = self.__create_list_with_slices__(self.ct_list, self.num_slices)
            self.seg_list_w_slices = self.__create_list_with_slices__(self.seg_list, self.num_slices)

        self.__length__ = len(self.ct_list_w_slices)

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
    
    def __create_list_w_slices_from_pickle__(self, path):
        """
        Create a list where each item is a tuple
        1st item in tuple is path to image and the 2nd is the desired slice.
        This function relies on a pickle file containing desired slices for each image
        @path - path to the pickle file
        """
        with open(path, 'rb') as handle:
            loaded_dict = pickle.load(handle)
        list_w_slices = [[tuple([key, item]) for item in slices] for key, slices in loaded_dict.items()]
        merged_list_w_slices = list(itertools.chain.from_iterable(list_w_slices))

        return merged_list_w_slices
    
    def __create_list_with_slices__(self, lst, n_slices):
        """
        create a list of tuples where the 1st item is path to ct
        and the 2nd item is the deisred slice
        """
        list_w_slices = [[tuple([name, i]) for i in range(n_slices)] for name in lst]
        merged_list = list(itertools.chain.from_iterable(list_w_slices))

        return merged_list 

    def __getitem__(self, index):
        example = {}

        ct_path, ct_slice = self.ct_list_w_slices[index]
        seg_path, seg_slice = self.seg_list_w_slices[index]
        # print(f'ct path:{ct_path} slice:{ct_slice}')
        # print(f'seg path:{seg_path} slice:{seg_slice}')
        
        ct = sitk.ReadImage(ct_path, sitk.sitkUInt8)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)[ct_slice]
        seg_array = sitk.GetArrayFromImage(seg)[seg_slice]

        ct_img = self.__convert_slice_to_img__(ct_array)
        # seg_img = self.__convert_slice_to_img__(seg_array)
        seg_img = self.__convert_segmap_to_rgb_hard__(seg_array)

        if self.tokenizer:
            example['instance_prompt_ids'] = self.tokenizer(
                self.prompt,
                padding='do_not_pad',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        example['instance_image'] = self.image_transforms(ct_img)
        example['instance_segmap_image'] = self.segmap_transforms(seg_img)

        return example

    def __len__(self):
        return self.__length__

    def __convert_slice_to_img__(self, slice):
        arr = slice
        if len(arr.shape) > 3:
            arr = arr.squeeze(0)
        # if arr.shape == 3:
        #     arr = np.moveaxis(arr, [0, 1, 2], [2, 0, 1])

        img = Image.fromarray(arr).convert('RGB')

        return img
    
    def __convert_segmap_to_rgb__(self, seg_arr):
        seg_arr = seg_arr.astype(np.uint16)
        seg_arr = seg_arr * 128
        seg_arr = Image.fromarray(seg_arr).convert('RGB')

        return seg_arr
    
    def __convert_segmap_to_rgb_hard__(self, seg_arr):
        seg_arr = seg_arr.astype(np.uint8)
        # create an empty image with the same shape as the input array
        image = np.zeros((seg_arr.shape[0], seg_arr.shape[1], 3), dtype=np.uint8)
        
        # convert 0 values to (0, 0, 0)
        image[seg_arr == 0] = [0, 0, 0]
        
        # convert 1 values to (128, 128, 128)
        image[seg_arr == 1] = [128, 128, 128]
        
        # convert 2 values to (255, 255, 255)
        image[seg_arr == 2] = [255, 255, 255]
        
        # convert the numpy array to a PIL image
        return Image.fromarray(image)
    
    def __convert_slices_to_images__(self, ct_arr):
        ct_imgs = []
        n_slices = len(ct_arr)
        arr = np.array(ct_arr, dtype=np.uint8)
        if len(arr.shape) > 3:
            arr = arr.squeeze(0)

        arr = np.moveaxis(arr, [0, 1, 2], [2, 0, 1])
        imgs = [Image.fromarray(arr[:,:,indx]).convert('RGB') for indx in range(n_slices)]

        return imgs