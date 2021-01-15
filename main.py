import json
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn.functional import one_hot
from torchvision import datasets, models, transforms
from collections import Counter
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from constants import GROUP_TYPES, CELEBA_ATTRIBUTE_NAMES
from hsic import HSIC
from configs import DATASET_DIR, EPOCHS, SHOW_TYPE_COUNTS, BATCH_SIZE, LEARNING_RATE, HSIC_WEIGHT, GPUS, \
    PBAR_REFRESH_RATE, LR_SCHEDULER_PATIENCE


class Resnet50Classifer(pl.LightningModule):
    #TODO add hyperparameters to constructor input!
    def __init__(self, *args, **kwargs):
        # init a pretrained resnet
        super().__init__(*args, **kwargs)
        num_target_classes = 2
        self.model = models.resnet50(pretrained=True)

        # use the pretrained model to classify
        self.model.fc = nn.Linear(self.model.fc.in_features, num_target_classes)

        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.batch_size = BATCH_SIZE

    def forward(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=LR_SCHEDULER_PATIENCE, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def setup(self, stage: str):
        def target_transform(y):
            target_names = ['Male', 'Blond_Hair']
            attr_dict = dict(zip(CELEBA_ATTRIBUTE_NAMES, y))
            attr_dict = {key: attr_dict[key] for key in target_names}
            attr_dict['group_idx'] = GROUP_TYPES[
                f'Male = {bool(attr_dict["Male"].item())}, Blond = {bool(attr_dict["Blond_Hair"].item())}']

            return attr_dict

        self.train_data = datasets.CelebA(DATASET_DIR, target_type='attr', split='train', download=False,
                                          target_transform=target_transform, transform=transforms.ToTensor())
        self.val_data = datasets.CelebA(DATASET_DIR, target_type='attr', split='valid', download=False,
                                        target_transform=target_transform, transform=transforms.ToTensor())
        if SHOW_TYPE_COUNTS:
            def count_types(data):
                counter = Counter()
                for x, y in data:
                    type = f'Male = {bool(y["Male"].item())}, Blond = {bool(y["Blond_Hair"].item())}'
                    counter[type] += 1

                print(json.dumps(counter, indent=4))

            print('Training data')
            count_types(self.train_data)
            print('\n')

            print('Validation data')
            count_types(self.val_data)
            print('\n')

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, pin_memory=True)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        return val_loader

    def step(self, batch, batch_idx):
        x, y = batch

        # 1 forward
        y_hat = self.forward(x)

        losses = self.loss(y_hat, y['Blond_Hair'])

        acc = accuracy(y_hat, y['Blond_Hair'])

        group_map = one_hot(y['group_idx'], num_classes=4).float()

        group_count = group_map.sum(0)
        n = group_count + (group_count == 0).float()  # avoid nans

        compute_group_avg = lambda m: ((group_map.t() @ m.view(-1).cuda()) / n)

        group_loss = compute_group_avg(losses)
        group_acc = compute_group_avg((torch.argmax(y_hat,1)==y['Blond_Hair']).float())

        cross_entropy = losses.mean()

        c = one_hot(y['group_idx'] % 2, num_classes=2).float()   # female to 1 0, male to
        hsic = HSIC(c, y_hat)

        loss = cross_entropy + HSIC_WEIGHT * hsic
        metrics = {
            'cross_entropy':cross_entropy,
            'hsic': hsic,

            'loss': loss,
            'loss_group_0': group_loss[0],
            'loss_group_1': group_loss[1],
            'loss_group_2': group_loss[2],
            'loss_group_3': group_loss[3],

            'acc': acc,
            'acc_group_0': group_acc[0],
            'acc_group_1': group_acc[1],
            'acc_group_2': group_acc[2],
            'acc_group_3': group_acc[3],
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        metrics = {f'train_{key}':metrics[key] for key in metrics}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        metrics = {f'val_{key}':metrics[key] for key in metrics}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss



model = Resnet50Classifer()

tb_logger = pl_loggers.TensorBoardLogger('logs/', version='hsic_50')



trainer = pl.Trainer(max_epochs=EPOCHS, gpus=GPUS, progress_bar_refresh_rate=PBAR_REFRESH_RATE, logger=tb_logger)


tb_logger.log_hyperparams({'BATCH_SIZE': BATCH_SIZE,
                           'LEARNING_RATE': LEARNING_RATE,
                           'HSIC_WEIGHT': HSIC_WEIGHT,
                           'LR_SCHEDULER': 'ReduceLROnPlateau',
                           'LR_SCHEDULER_PATIENCE': LR_SCHEDULER_PATIENCE})

trainer.fit(model)
