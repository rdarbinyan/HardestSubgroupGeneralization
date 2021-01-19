import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import logging
from torch.nn.functional import one_hot
from pytorch_lightning.metrics.functional import accuracy
from omegaconf import DictConfig, OmegaConf

import utils
from src.config_parser.train import ConfigParser
from src.utils.hsic import HSIC

# TODO move this to hydra
HSIC_WEIGHT = 10

log = logging.getLogger(__name__)

class Model(pl.LightningModule, ConfigParser):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.config = self.parse_config(cfg)

        self.network = self.config.network.get_network()

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        outputs = self.network(x)
        return outputs

    def configure_optimizers(self):
        optimizer = self.config.optimizer.get_optimizer(self.parameters())
        scheduler = self.config.scheduler.get_scheduler(optimizer)

        ret_opt = {"optimizer": optimizer}

        if scheduler is not None:
            sch_opt = {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }

            ret_opt.update({"lr_scheduler": sch_opt})

        return ret_opt

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

@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    log.info(f"\nGiven Config:\n {OmegaConf.to_yaml(cfg)}")

    config = ConfigParser.parse_config(cfg)

    log.info(f"\nResolved Dataclass:\n {config} \n")

    utils.fix_seeds(config.random_seed)

    logger = config.logger.get_logger(config.logs_root_dir)
    datamodule = config.dataset.get_datamodule()

    model = Model(cfg)

    trainer = config.trainer.get_trainer(logger, config.logs_root_dir)

    trainer.fit(model, datamodule=datamodule)



if __name__ == '__main__':
    main()