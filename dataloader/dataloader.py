import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import WeightedRandomSampler

from PIL import Image

from dataloader.nir_preprocessing import get_covars_from_data
import random


def jpeg_compression(rgb):
    device = rgb.device
    rgb = rgb.mul(255).to(torch.uint8).cpu()
    if torch.rand(1).item() < 0.5:
        rgb = T.JPEG(quality=10)(rgb)
    rgb = rgb.to(torch.float32).div(255).to(device)
    return rgb

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text
    sort_key = lambda key: [ convert(num) for num in re.split(r'(\d+)', key)]
    l.sort(key=sort_key)
    return l


class FireDataset(torch.utils.data.Dataset):
    def __init__(self, opts, data_type, device='cuda', augment=True, val_range=None):
        super(FireDataset, self).__init__()
        self.classes = [10, 100, 0]
        self.num_per_class = [46, 794, 256] # 256, 46, 794
        self.opts = opts
        self.data_dir  = os.path.join(opts.data_dir, 'Images')
        self.data_type = data_type
        self.file_list = os.listdir(self.data_dir)
        self.device = device
        self.augment = augment
        self.standard_augment = opts.standard_augment
        self.gaussian_nir = opts.gaussian_nir

        self.file_list = natural_sort(self.file_list)

        annotations_path = os.path.join(opts.data_dir, 'labels.csv')
        self.annotations_data = np.genfromtxt(annotations_path, delimiter=',')

        # Augmentation Pipeline
        trans_list = [
            T.Resize(int(self.opts.image_size*1.8)),
            T.RandomHorizontalFlip(p=0.5), 
            T.RandomVerticalFlip(p=0.5), 
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(self.opts.image_size, (0.6, 1.2)),
        ]
        
        if self.opts.cutout:
            trans_list.append(T.RandomErasing(p=0.5,scale=(0.02,0.2),ratio=(0.3,3.3),value="random"))

        # RGB augmentations
        rgb_aug = []
        rgb_nir_aug = []
        if self.opts.jpeg:
            rgb_aug.extend([
                T.Lambda(jpeg_compression)])
        if self.opts.noise:
            rgb_aug.append(T.GaussianNoise())
        if self.opts.blur:
            rgb_aug.append(T.GaussianBlur(kernel_size=5))
            rgb_nir_aug.append(T.GaussianBlur(kernel_size=5)) 
        self.augmentations = T.Compose(trans_list)
        self.rgb_nir_augmentations = T.Compose(rgb_nir_aug)
        self.rgb_augmentations = T.Compose(rgb_aug)

        self.default_transform = T.Compose([
            T.Resize(self.opts.image_size),
            T.CenterCrop(self.opts.image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])

        self.annotations = []
        self.data = []
        for class_idx, num_data in enumerate(self.num_per_class):
            # print(class_idx, num_data)
            relative_start_idx = []
            relative_end_idx = []
            if val_range is None:
                if data_type == 'train':
                    relative_start_idx.append(0)
                    relative_end_idx.append(int(num_data * opts.test_train_split[0]))
                elif data_type == 'validate':
                    relative_start_idx.append(int(num_data * opts.test_train_split[0]))
                    relative_end_idx.append(int(num_data * (opts.test_train_split[0] + opts.test_train_split[1])))
                elif data_type == 'test':
                    relative_start_idx.append(int(num_data * (opts.test_train_split[0] + opts.test_train_split[1])))
                    relative_end_idx.append(num_data)
                elif data_type == "all":
                    relative_start_idx = 0
                    relative_end_idx = num_data
            else:
                if data_type == 'validate':
                    relative_start_idx.append(int(num_data * val_range[0]))
                    relative_end_idx.append(int(num_data * val_range[1]))
                else:
                    relative_start_idx.append(0)
                    relative_end_idx.append(int(num_data*val_range[0]))
                    relative_start_idx.append(int(num_data * val_range[1]))
                    relative_end_idx.append(int(num_data * val_range[2]))
            #print(class_idx, relative_start_idx, relative_end_idx)
            self.get_class_data(class_idx, relative_start_idx, relative_end_idx)
        
        self.training = False
        if data_type == "train":
            self.training = True 
            self.get_covars()
            
        
        
    # 
    def get_class_data(self, class_idx, relative_idx_start, relative_idx_end):
        initial_offset = 0
        for i in range(class_idx):
            initial_offset += self.num_per_class[i]

        file_names = []
        csv_annotations = []
        for i in range(len(relative_idx_start)):
            absolute_index_start = initial_offset + relative_idx_start[i]
            absolute_index_end = initial_offset + relative_idx_end[i]
            total_data = absolute_index_end - absolute_index_start
            # print(class_idx, absolute_index_start, absolute_index_end)

            file_names.extend(self.file_list[absolute_index_start:absolute_index_end])
            csv_annotations.extend(self.annotations_data[1+absolute_index_start:1+absolute_index_end,2])

        
        for i, zipped in enumerate(zip(file_names, csv_annotations)):
            file_name, annotation = zipped
            # print(file_name, annotation)
            
            image = Image.open(os.path.join(self.data_dir, file_name))
            
            img_data = self.preprocessing(image)
            self.data.append(img_data)

            annotation = int(annotation)
            one_hot_ann = [0,0,0]
            if annotation == 0:
                one_hot_ann[0] = 1
            elif annotation == 10:
                one_hot_ann[1] = 1
            elif annotation == 100: 
                one_hot_ann[2] = 1
            else:
                raise ValueError(f"Unexpected class annotation - \"{annotation}\" encountered")
            
            self.annotations.append(one_hot_ann)

    
    # Get the class covariances to sample from an MVN for data augmentation        
    def get_covars(self):
        my_file = Path("dataloader/covars")
        if my_file.is_dir():
            covar_0 = np.load("dataloader/covars/class_0_cov.npy")
            covar_1 = np.load("dataloader/covars/class_1_cov.npy")
            covar_2 = np.load("dataloader/covars/class_2_cov.npy")
        else:
            os.makedirs("dataloader/covars")
            covar_0, covar_1, covar_2 = get_covars_from_data(self.data, self.annotations)
            np.save("dataloader/covars/class_0_cov.npy", covar_0)
            np.save("dataloader/covars/class_1_cov.npy", covar_1)
            np.save("dataloader/covars/class_2_cov.npy", covar_2)

        covar_0 = torch.from_numpy(covar_0).to(self.device)
        covar_1 = torch.from_numpy(covar_1).to(self.device)
        covar_2 = torch.from_numpy(covar_2).to(self.device)
        #print(covar_0.dtype)

        n0, _ = covar_0.shape
        n1, _ = covar_1.shape
        n2, _ = covar_2.shape

        mean0 = torch.zeros((n0), dtype=torch.float64).to(self.device)
        #print(mean0.dtype)
        self.MVN_0 = MultivariateNormal(loc = mean0, covariance_matrix = covar_0)

        mean1 = torch.zeros((n1), dtype=torch.float64).to(self.device)
        self.MVN_1 = MultivariateNormal(loc = mean1, covariance_matrix = covar_1)
        
        mean2 = torch.zeros((n2), dtype=torch.float64).to(self.device)
        self.MVN_2 = MultivariateNormal(loc = mean2, covariance_matrix = covar_2)

                        


    def preprocessing(self, image):
        
        self.transforms = transforms.Compose([
            # transforms.Resize(self.opts.image_size),
            # transforms.CenterCrop(self.opts.image_size), # some images are "long" for some reason instead of square look into this
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])

        self.normalize_nir = transforms.Normalize((0.5), (0.5))
        image = self.transforms(image)
      
        return image

    def augmentation_pipeline(self, image, ann):
        image = self.augmentations(image)

        # RandomBrightnessContrast on RGB+NIR
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            bias = random.uniform(-0.05, 0.05)
            image = image * scale + bias
            image = torch.clamp(image, 0, 1)

        # Channel-wise dropout
        if random.random() < 0.15:
            if random.random() < 0.5:
                image[:3] = 0  # Drop RGB
            else:
                image[3:] = 0  # Drop NIR

        if self.gaussian_nir:
            image = self.add_NIR_noise(image, ann)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_data = self.data[idx].to(self.device)
        annotation_data = torch.FloatTensor(self.annotations[idx])
        if self.augment:
            if self.standard_augment:
                rgb, nir = im_data[:3], im_data[3:]
                rgb = self.rgb_augmentations(rgb)
                if not self.opts.split_rgb_and_nir:
                    #print(" RGB and NIR are not split!")
                    if self.gaussian_nir:
                        nir = self.rgb_nir_augmentations(nir)
                    else:
                        nir = self.rgb_augmentations(nir)
                im_data = torch.cat((rgb, nir), dim=0)
            if self.gaussian_nir:
                self.add_NIR_noise(im_data, annotation_data)
            im_data = self.augmentation_pipeline(im_data, annotation_data)
        im_data = self.default_transform(im_data)
        return im_data.to(self.device), annotation_data.to(self.device)

    def add_NIR_noise(self, im, ann):
        cls = torch.argmax(ann)
        if cls == 0:
            MVN = self.MVN_0
        elif cls == 1:
            MVN = self.MVN_1
        elif cls == 2:
            MVN = self.MVN_2
        #print("im shape", im.shape)
        sample = MVN.rsample(sample_shape=im[3,:,:].shape)
        idx = random.randint(0,self.opts.NIR_buckets -1)
        noise = sample[:,:,idx]
        im[3,:,:]  =  im[3,:,:] + noise
        return im



def get_dataloader(opts, data_type, shuffle=False, augment=False, sample=False, kfold_val=None):
    dataset = FireDataset(opts, data_type, augment=augment)
    if sample:
        targets = torch.tensor([ann.index(1) for ann in dataset.annotations]) # produces [0,2,1,1,0, ...] (reverses one-hot)
        
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float() # Get the class probabilities.

        # Get weight for each sample.
        sample_weights = class_weights[targets]

        sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

        dataloader =  torch.utils.data.DataLoader(dataset, opts.batch_size, sampler=sampler, shuffle=shuffle)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, opts.batch_size, shuffle=shuffle)
    

    return dataloader


