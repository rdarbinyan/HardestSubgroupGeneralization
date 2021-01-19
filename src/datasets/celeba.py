import logging
from collections import Counter

import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

log = logging.getLogger(__name__)

# TODO try to get rid of this
CELEBA_ATTRIBUTE_NAMES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                          'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                          'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                          'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                          'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                          'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


class CelebADataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root: str,
                 batch_size: int,
                 num_workers: int,
                 confounder_name: str,
                 target_name: str,
                 count_types_and_log: bool):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.confounder_name = confounder_name
        self.target_name = target_name

        self.count_types_and_log = count_types_and_log

    def prepare_data(self, *args, **kwargs):
        def target_transform(y):
            target_names = [self.confounder_name, self.target_name]
            attr_dict = dict(zip(CELEBA_ATTRIBUTE_NAMES, y))
            attr_dict = {key: attr_dict[key] for key in target_names}
            attr_dict['group_idx'] = attr_dict[self.confounder_name] + 2 * attr_dict[self.target_name]  # binary pair (confounder, target) -> decimal number

            return attr_dict

        self.train_data = datasets.CelebA(self.data_root, target_type='attr', split='train', download=False,
                                          target_transform=target_transform, transform=transforms.ToTensor())
        self.val_data = datasets.CelebA(self.data_root, target_type='attr', split='valid', download=False,
                                        target_transform=target_transform, transform=transforms.ToTensor())
        if self.count_types_and_log:
            def count_types(data):
                counter = Counter()
                for x, y in data:
                    type = f'{self.confounder_name} = {bool(y[self.confounder_name].item())}, {self.target_name} = {bool(y[self.target_name].item())}'
                    counter[type] += 1

                return json.dumps(counter, indent=4)

            log.info('counting types...')

            log.info(f'Training data \n {count_types(self.train_data)}')

            log.info(f'Validation data \n {count_types(self.val_data)}')

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        return val_loader
