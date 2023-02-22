from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os
from data.get_voc_color_by_class import get_voc_colormap_png, color_map, get_color_map_by_class_rgb
import xml.etree.ElementTree as ET
import numpy as np
import torch
import itertools

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        img_transforms = []

        if resize:
            img_transforms.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        if self.tokenizer:
            example["instance_prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            if self.tokenizer:
                example["class_prompt_ids"] = self.tokenizer(
                    self.class_prompt,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).input_ids

        return example

class SegMapDataset(DreamBoothDataset):

    def __init__(self,
        instance_data_root,
        instance_segmap_data_root,
        annotations_root,
        instance_prompt,
        tokenizer=None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
    ):
        super().__init__(instance_data_root,
                        instance_prompt,
                        tokenizer,
                        class_data_root,
                        class_prompt,
                        size,
                        center_crop,
                        color_jitter,
                        resize=resize)

        # self.instance_segmap_images_path = list(Path(instance_segmap_data_root).iterdir())
        self.instance_segmap_images_path = [os.path.join(instance_segmap_data_root, os.path.splitext(os.path.basename(path))[0] + '.png') for path in self.instance_images_path]
        # sanity check
        assert len(self.instance_segmap_images_path) == len(self.instance_images_path), "number of segmaps is different from number of images!"

        self.instance_segmap_data_root = instance_segmap_data_root
        self.annotations_root = annotations_root
        self.class_annotations = self.__get_class_annotations__()
        
        # duplicate paths in instance_segmap_data_root to match the number of class images in class annotations
        self.instance_segmap_images_path = self.__duplicate_paths_according_to_annotations__(self.instance_segmap_images_path, self.class_annotations)
        self.instance_images_path = self.__duplicate_paths_according_to_annotations__(self.instance_images_path, self.class_annotations)
        
        # combine all lists in class_annotations to a single list
        merged = list(itertools.chain(*self.class_annotations))
        self.class_annotations = merged
        assert len(self.class_annotations) == len(self.instance_segmap_images_path) == len(self.instance_images_path), \
            "number of class annotations is different from number of segmaps or images!"
        
        self.__data_len__ = len(self.instance_segmap_images_path)
        
        self.voc_png_colormap_dict = get_voc_colormap_png()
        self.voc_rgb_colormap_dict = get_color_map_by_class_rgb()
        self.voc_colormap = color_map()

        img_transforms = []
        segmap_transforms = []

        if resize:
            resize_t = transforms.Resize(
                    (size,size), interpolation=transforms.InterpolationMode.BILINEAR
                )
            img_transforms.append(resize_t)
            segmap_transforms.append(transforms.Resize(
                    (size, size), interpolation=transforms.InterpolationMode.NEAREST
                )
            )
        
        if center_crop:
            center_crop_t = transforms.CenterCrop(size)
            img_transforms.append(center_crop_t)
            segmap_transforms.append(center_crop_t)

        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        
        if h_flip:
            h_flip_t = transforms.RandomHorizontalFlip()
            img_transforms.append(h_flip_t)
            segmap_transforms.append(h_flip_t)

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
            ]
        )
        self.segmap_transforms = transforms.Compose(
            [*segmap_transforms]
        )
        
    '''
    self.annotations should be list of lists at this point
    duplicate each image path in instance images and segmap images to match the number of classes in each correspinding place in class annotations
    '''
    def __duplicate_paths_according_to_annotations__(self, img_list, annotations):
        tmp_img_list = []
        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                tmp_img_list.append(img_list[i])
        return tmp_img_list
        
    
    '''
    return list of lists of class annotations for each image
    that is, each image has a list will all the classes in it
    '''
    def __get_class_annotations__(self):
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

        images_names = [os.path.splitext(os.path.basename(path))[0] for path in self.instance_segmap_images_path]
        xmls_paths = [os.path.join(self.annotations_root, img_name +'.xml') for img_name in images_names]
    
        annotations = []
        for path in xmls_paths:
            tree = ET.parse(path).getroot()
            objects = tree.findall('object/name')
            annotations.append(list(set(
                [voc_classes[object.text] for object in objects if object.text in voc_classes.keys()]
                )))

        return annotations

    def __get_png_segmap_by_class__(self, img, voc_class):
        seg_img = np.asarray(img)
        
        voc_class_pixel_val = self.voc_png_colormap_dict[voc_class]
        void_pixel_val = self.voc_png_colormap_dict['void']
        bg_pixel_val = self.voc_png_colormap_dict['background']

        tmp_img = seg_img.copy()
        tmp_img[tmp_img != voc_class_pixel_val] = bg_pixel_val

        tmp_img = Image.fromarray(tmp_img.astype('uint8'))
        tmp_img = tmp_img.convert('P')
        tmp_img.putpalette(self.voc_colormap)

        return tmp_img

    def __get_rgb_segmap_by_class__(self, img, voc_class):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        seg_img = np.asarray(img)
        new_seg_img = seg_img.copy()

        new_seg_img[~np.all(new_seg_img == self.voc_rgb_colormap_dict[voc_class], axis=-1)] = [0, 0, 0]
        new_seg_img = Image.fromarray(new_seg_img)

        if new_seg_img.mode != 'RGB':
            new_seg_img = new_seg_img.convert('RGB')

        return new_seg_img
    
    '''
        Gets png image of shape (h,w) and voc_class
        Returns a torch tensor where all pixel that arent the class are (0,0,0) and (255,255,255) for the class
    '''
    def __get_bin_segmap_by_class__(self, img, voc_class):
        seg_img = np.asarray(img)
        new_seg_img = seg_img.copy()

        new_seg_img[new_seg_img != self.voc_png_colormap_dict[voc_class]] = 0
        new_seg_img[new_seg_img == self.voc_png_colormap_dict[voc_class]] = 255

        new_seg_img = np.repeat(new_seg_img[:, :, np.newaxis], 3, axis=2)
        new_seg_img = torch.from_numpy(new_seg_img).permute(2, 0, 1)

        return new_seg_img

    
    def __getitem__(self, index):
        example = {}
        img_path = self.instance_images_path[index % self.__data_len__]
        _, img_name = os.path.split(img_path)
        instance_image = Image.open(
            img_path
        )

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        ## Segmap
        segmap_img_path = self.instance_segmap_images_path[index % self.__data_len__]
        segmap_instance_image = Image.open(segmap_img_path)
        example['instance_images_clean'] = np.asarray(segmap_instance_image)
        
        segmap_instance_image = self.segmap_transforms(segmap_instance_image)

        voc_class = self.class_annotations[index % self.__data_len__]
        # segmap_instance_image.save(f'{voc_class}_before.jpeg')
        
        segmap_instance_tensor = self.__get_bin_segmap_by_class__(segmap_instance_image, voc_class)
        # segmap_instance_image.save(f'{voc_class}.jpeg')

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_segmap_images"] = segmap_instance_tensor
        example["instance_classes"] = voc_class
        
        if self.tokenizer:
            example["instance_prompt_ids"] = self.tokenizer(
                self.instance_prompt + " " + voc_class,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        # print(f'mask:{segmap_img_path}, annotation:{self.class_annotations[index % self.num_instance_images]}')

        return example
    
    def __len__(self):
        return self.__data_len__
