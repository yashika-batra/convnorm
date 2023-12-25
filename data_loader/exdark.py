import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

exDarkClasses = {
            "Bicycle": 0,
            "Boat": 1,
            "Bottle": 2,
            "Bus": 3,
            "Car": 4,
            "Cat": 5,
            "Chair": 6,
            "Cup": 7,
            "Dog": 8,
            "Motorbike": 9,
            "People": 10,
            "Table": 11
        }

class ExDarkData_NoTransforms(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.classes = exDarkClasses

    def __len__(self):
        return len(list(self.labels))

    def __getitem__(self, idx):
        # loop through all classes to get the index of the right class
        image = Image.fromarray(self.images[idx] * 255, mode='RGB')
        label = self.labels[idx]
        return image, label


def get_exdark(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True):
    # base_dataset = unpickle the root folder
    with open(root, "rb") as f:
        datadict = pickle.load(f)
        exdark_images = datadict["images"]
        exdark_labels = datadict["labels"]
        exdark_images = np.transpose(exdark_images, (0, 2, 3, 1))

    if train:
        # split data using Dataset class
        exdarkData_noTransforms = ExDarkData_NoTransforms(exdark_images, exdark_labels)
        trainSize = int(0.9 * len(exdarkData_noTransforms))
        testSize = len(exdarkData_noTransforms) - trainSize
        trainDataset, valDataset = torch.utils.data.random_split(exdarkData_noTransforms, [trainSize, testSize])
        
        # get train_images, train_labels, val_images, val_labels from Datasets
        train_idx = trainDataset.indices
        val_idx = valDataset.indices

        # create train and test datasets
        train_dataset = ExDark_Dataset(exdark_images[train_idx], exdark_labels[train_idx], transform_train)
        val_dataset = ExDark_Dataset(exdark_images[val_idx], exdark_labels[val_idx], transform_val)

        # inject noise
        if cfg_trainer['sym']:
            symmetric_noise(cfg_trainer, train_dataset)
            symmetric_noise(cfg_trainer, val_dataset)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000

    else:
        train_dataset = []
        val_dataset = ExDark_Dataset(exdark_images, exdark_labels, transform_val)
        print(f"Test: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def symmetric_noise(cfg_trainer, dataset):
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(dataset.train_data))
        for i, idx in enumerate(indices):
            if i < cfg_trainer['percent'] * len(dataset.train_data):
                dataset.noise_indx.append(idx)
                dataset.train_labels[idx] = np.random.randint(dataset.num_classes, dtype=np.int32)


class ExDark_Dataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.train_data = images
        self.train_labels = labels
        self.classes = exDarkClasses
        self.num_classes = len(list(exDarkClasses.keys()))
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
