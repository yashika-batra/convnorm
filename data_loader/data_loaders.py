import sys

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.cifar10 import get_cifar10
from data_loader.coco import get_coco
from data_loader.exdark import get_exdark
# from data_loader.xx_cifar100 import get_cifar100
from parse_config import ConfigParser
from PIL import Image
import pickle
import numpy as np

class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        if cfg_trainer["do_adv"]:
            print("Doint adv. attack")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        
        
        self.data_dir = data_dir

        # noise_file='%sCIFAR10_%.1f_Asym_%s.json'%(config['data_loader']['args']['data_dir'],cfg_trainer['percent'],cfg_trainer['asym'])
        
        self.train_dataset, self.val_dataset = get_cifar10(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val)#, noise_file = noise_file)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)


# add a dataloader for tinyCOCO
class COCODataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        size = 224

        # update these transformations based off what the norm values are for COCO
        # take the mean over each channel and std over each channel
        with open(data_dir, "rb") as f:
            datadict = pickle.load(f)
            coco_images = datadict["images"]
            # find means and stds per channel
            print(coco_images.shape)
            (size, h, w, c) = coco_images.shape
            flat_wrt_channels = np.reshape(np.transpose(coco_images, (3, 0, 1, 2)), (c, size * h * w))
            means, stds = list(np.mean(flat_wrt_channels, axis=1)), list(np.std(flat_wrt_channels, axis=1))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_val = transforms.Compose([
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        
        self.data_dir = data_dir
        
        self.train_dataset, self.val_dataset = get_coco(data_dir, cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)


# add a dataloader for ExDark
class ExDarkDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        size = 224

        with open(data_dir, "rb") as f:
            datadict = pickle.load(f)
            exdark_images = datadict["images"]
            # find means and stds per channel
            (size, c, h, w) = exdark_images.shape
            flat_wrt_channels = np.reshape(np.transpose(exdark_images, (1, 0, 2, 3)), (c, size * h * w))
            means, stds = list(np.mean(flat_wrt_channels, axis=1)), list(np.std(flat_wrt_channels, axis=1))

        # update these transformations based off what the norm values are for ExDark
        # take the mean over each channel and std over each channel
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_val = transforms.Compose([
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        
        self.data_dir = data_dir
        
        self.train_dataset, self.val_dataset = get_exdark(data_dir, cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)