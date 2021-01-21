import logging
import torch
from collections import Counter

import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


class CelebADataModule(pl.LightningDataModule):
    # TODO try to get rid of this
    CELEBA_ATTRIBUTE_NAMES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                              'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair','Bushy_Eyebrows',
                              'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                              'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                              'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    def __init__(self,
                 data_root: str,
                 download: bool,
                 batch_size: int,
                 num_workers: int,
                 confounder_name: str,
                 target_name: str,
                 sampler: str):
        super().__init__()
        self.data_root = data_root
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.confounder_name = confounder_name
        self.target_name = target_name

        self.sampler = sampler

    def __target_transform(self, y):
        target_names = [self.confounder_name, self.target_name]
        attr_dict = dict(zip(self.CELEBA_ATTRIBUTE_NAMES, y))
        attr_dict = {key: attr_dict[key] for key in target_names}
        attr_dict['group_idx'] = attr_dict[self.confounder_name] + 2 * attr_dict[self.target_name]  # binary pair (confounder, target) -> decimal number

        return attr_dict

    def __calculate_group_counts(self, dataset):
        attr_dict = dict(zip(self.CELEBA_ATTRIBUTE_NAMES, dataset.attr.T))
        group_indices = attr_dict[self.confounder_name] + 2 * attr_dict[self.target_name]

        group_counts = (torch.arange(4).unsqueeze(1) == group_indices).sum(1).float()

        return group_counts, group_indices

    def __set_group_counts(self):
        self.train_group_counts, self.train_group_indices = self.__calculate_group_counts(self.train_data)
        self.val_group_counts, self.val_group_indices = self.__calculate_group_counts(self.val_data)

        log.info(f'Training Group Counts \n {self.train_group_counts}')
        log.info(f'Validation Group Counts \n {self.val_group_counts}')

    def prepare_data(self, *args, **kwargs):
        self.train_data = datasets.CelebA(self.data_root,
                                          target_type='attr',
                                          split='train',
                                          download=self.download,
                                          target_transform=self.__target_transform,
                                          transform=transforms.ToTensor())

        self.val_data = datasets.CelebA(self.data_root,
                                        target_type='attr',
                                        split='valid',
                                        download=self.download,
                                        target_transform=self.__target_transform,
                                        transform=transforms.ToTensor())

        self.__set_group_counts()

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.sampler == "weighted":
            group_weights = len(self.train_data) / self.train_group_counts
            weights = group_weights[self.train_group_indices]
            sampler = WeightedRandomSampler(weights, len(self.train_data), replacement=True)

        train_loader = DataLoader(self.train_data,
                                  sampler=sampler,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self.val_data,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True)

        return val_loader
