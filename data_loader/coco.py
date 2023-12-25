import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pickle

CLASS_IDS = {
            'cat': 0,
            'dog': 1,
            'car': 2,
            'bus': 3,
            'train': 4,
            'boat': 5,
            'ball': 6,
            'pizza': 7,
            'chair': 8,
            'table': 9
        }

class COCOData_NoTransforms(Dataset):
    def __init__(self, sCOCO_images, sCOCO_labels):
        self.images = sCOCO_images
        self.labels = sCOCO_labels
        self.classes = CLASS_IDS
        self.size = 224
        print((self.images).shape, (self.labels).shape)

    def __len__(self):
        return len(list(self.labels))

    def __getitem__(self, idx):
        # loop through all classes to get the index of the right class
        image = Image.fromarray(self.images[idx] * 255, mode='RGB')
        label = self.labels[idx]
        return image, label


def get_coco(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True):
    # base_dataset = unpickle the root folder
    with open(root, "rb") as f:
        datadict = pickle.load(f)
        coco_images = datadict["images"]
        coco_labels = datadict["labels"]

    if train:
        # split data using Dataset class
        cocoData_noTransforms = COCOData_NoTransforms(coco_images, coco_labels)
        trainSize = int(0.9 * len(cocoData_noTransforms))
        testSize = len(cocoData_noTransforms) - trainSize
        trainDataset, valDataset = torch.utils.data.random_split(cocoData_noTransforms, [trainSize, testSize])
        
        # get train_images, train_labels, val_images, val_labels from Datasets
        train_idx = trainDataset.indices
        val_idx = valDataset.indices

        # create train and test datasets
        train_dataset = COCO_Dataset(coco_images[train_idx], coco_labels[train_idx], transform_train)
        val_dataset = COCO_Dataset(coco_images[val_idx], coco_labels[val_idx], transform_val)

        # inject noise
        if cfg_trainer['sym']:
            symmetric_noise(cfg_trainer, train_dataset)
            symmetric_noise(cfg_trainer, val_dataset)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000

    else:
        train_dataset = []
        val_dataset = COCO_Dataset(coco_images, coco_labels, transform_val)
        print(f"Test: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def symmetric_noise(cfg_trainer, dataset):
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(dataset.train_data))
        for i, idx in enumerate(indices):
            if i < cfg_trainer['percent'] * len(dataset.train_data):
                dataset.noise_indx.append(idx)
                dataset.train_labels[idx] = np.random.randint(dataset.num_classes, dtype=np.int32)


class COCO_Dataset(Dataset):
    def __init__(self, coco_images, coco_labels, transforms):
        self.train_data = coco_images
        self.train_labels = coco_labels
        self.classes = CLASS_IDS
        self.num_classes = 10
        self.transforms = transforms
        self.noise_indx = []

    def __len__(self):
        return len(list(self.train_labels))

    def __getitem__(self, idx):
        # loop through all classes to get the index of the right class
        image = Image.fromarray(self.train_data[idx] * 255, mode='RGB')
        image = self.transforms(image)
        label = self.train_labels[idx]
        return image, label
