import os
import torch.utils.data as data
import numpy as np
import PIL
from path import Path
import random
from PIL import Image

class SequenceFolder(data.Dataset):

    def __init__(self, root, transform=None, seed=None, train=True, grayscale=False, hsv=False):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.grayscale=grayscale
        self.hsv=hsv
        self.crawl_folders(train)

    def crawl_folders(self, train):
        sequence_set = []
        for scene in self.scenes:
            file = open(scene+"/triplets.txt", "r")
            for line in file:
                if line=='\n':
                    continue
                tokens=line.strip().split('@')
                sample = {'anchor': os.path.join(scene,'data',tokens[0]),
                          'positive': os.path.join(scene,'data',tokens[1]),
                          'apbox':tokens[2], 'nbox':tokens[3] }
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        apbox=[int(s) for s in sample['apbox'].split(',') if s.isdigit()]
        nbox=[int(s) for s in sample['nbox'].split(',') if s.isdigit()]
        if self.grayscale:
            anchor_img = Image.open(sample['anchor']).convert('L').crop(apbox)
            positive_img = Image.open(sample['positive']).convert('L').crop(apbox)
            negative_img = Image.open(sample['positive']).convert('L').crop(nbox)
        elif self.hsv:
            anchor_img = Image.open(sample['anchor']).convert('HSV').crop(apbox)
            positive_img = Image.open(sample['positive']).convert('HSV').crop(apbox)
            negative_img = Image.open(sample['positive']).convert('HSV').crop(nbox)
        else:
            anchor_img = Image.open(sample['anchor']).crop(apbox)
            positive_img = Image.open(sample['positive']).crop(apbox)
            negative_img = Image.open(sample['positive']).crop(nbox)
        imgs=[anchor_img, positive_img, negative_img]

        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs



    def __len__(self):
        return len(self.samples)

class SequenceFolder_Validation_Full(data.Dataset):

    def __init__(self, root, crop_size=(0,0),resize_ratio=0, grayscale=False, hsv=False, transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crop_size = crop_size
        self.resize_ratio=resize_ratio
        self.grayscale=grayscale
        self.hsv=hsv
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            if 'VAL' in scene:
                scene_ims = os.listdir(os.path.join(scene))
                sample = {'anchor': os.path.join(scene,scene_ims[0]),
                          'positive': os.path.join(scene, scene_ims[1])}
                if sample not in sequence_set:
                    sequence_set.append(sample)
        self.samples = sequence_set


    def __getitem__(self, index):
        sample = self.samples[index]
        anchor_img = Image.open(sample['anchor'])
        posisitve_img = Image.open(sample['positive'])
        if self.crop_size[0]>0:
            original_size=anchor_img.size
            crop_box=[(original_size[0]-self.crop_size[0])//2,(original_size[1]-self.crop_size[1])//2,
                      (original_size[0]+self.crop_size[0])//2,(original_size[1]+self.crop_size[1])//2]
            # crop to given size
            anchor_img = anchor_img.crop(crop_box)
            posisitve_img = posisitve_img.crop(crop_box)
        if self.resize_ratio>0:
            anchor_img=anchor_img.resize((np.array(anchor_img.size)/self.resize_ratio).astype(np.int),PIL.Image.BILINEAR)
            posisitve_img=posisitve_img.resize((np.array(posisitve_img.size)/self.resize_ratio).astype(np.int),PIL.Image.BILINEAR)
        imgs=[anchor_img,posisitve_img]
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs

    def __len__(self):
        return len(self.samples)

class SequenceFolder_Inference(data.Dataset):

    def __init__(self, root, crop_size=(0,0),resize_ratio=0, transform=None):
        self.root = Path(root)
        scene_list_path = self.root / 'val.txt'
        self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crop_size = crop_size
        self.resize_ratio=resize_ratio
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            if 'VAL' in scene or 'TEST' in scene or 'perfect' in scene:
                for image in os.listdir(os.path.join(scene)):
                    if 'png' in image or 'jpg' in image:
                        sample = {'image': os.path.join(scene,image)}
                        sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        anchor_img = Image.open(sample['image'])
        if self.crop_size[0]>0:
            crop_size_0=np.min((self.crop_size[0],anchor_img.size[0]))
            crop_size_1=np.min((self.crop_size[1],anchor_img.size[1]))
            anchor_img = anchor_img.crop([0,0,crop_size_0,crop_size_1])
        if self.resize_ratio>0:
            anchor_img=anchor_img.resize((np.array(anchor_img.size)/self.resize_ratio).astype(np.int),PIL.Image.BILINEAR)
        imgs=[anchor_img]
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, sample['image']

    def __len__(self):
        return len(self.samples)

class SequenceFolderPlane(data.Dataset):

    def __init__(self, root, crop_size=(0,0),resize_ratio=0, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.crop_size = crop_size
        self.resize_ratio=resize_ratio
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        for image in os.listdir(self.root):
            sample = {'image': os.path.join(self.root, image)}
            sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        anchor_img = Image.open(sample['image']).convert('RGB')
        imgs=[anchor_img]
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, sample['image']

    def __len__(self):
        return len(self.samples)
