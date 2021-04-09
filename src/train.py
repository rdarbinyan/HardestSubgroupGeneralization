import sys, os
from typing import List, Any

sys.path.append(os.path.dirname(sys.path[0]))

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

log = logging.getLogger(__name__)


class Model(pl.LightningModule, ConfigParser):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.config = self.parse_config(cfg)

        self.network = self.config.network.get_network()

        self.__cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.__hsic_weight = 0

        self.__group_weights = self.__get_initial_group_weights(4)  # sets (1/n, 1/n, 1/n, 1/n)

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

    def training_step(self, batch, *args, **kwargs):
        return self.__step(batch)

    def validation_step(self, batch, *args, **kwargs):
        return self.__step(batch)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        epoch_metrics = self.__calculate_epoch_metrics(outputs)
        epoch_metrics = {f'train_{key}': epoch_metrics[key] for key in epoch_metrics}

        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        epoch_metrics = self.__calculate_epoch_metrics(outputs)
        epoch_metrics = {f'val_{key}': epoch_metrics[key] for key in epoch_metrics}

        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)

    @staticmethod
    def __get_initial_group_weights(groups_count):
        group_weights = torch.ones(groups_count)
        group_weights = group_weights / group_weights.sum()
        group_weights = group_weights.to('cpu')

        return group_weights

    @staticmethod
    def __get_group_metrics(y, y_hat, group_indices, cross_entropies):
        group_map = one_hot(group_indices, num_classes=4).float()

        group_counts = group_map.sum(0)
        n = group_counts + (group_counts == 0).float()  # avoid nans

        compute_group_avg = lambda m: ((group_map.t() @ m.view(-1).cuda()) / n)

        group_cross_entropy = compute_group_avg(cross_entropies)
        group_acc = compute_group_avg((torch.argmax(y_hat, 1) == y).float())

        return group_cross_entropy, group_acc, group_counts

    def __update_hsic_weight(self):
        if self.config.hsic.name == "constant_weight":
            self.__hsic_weight = self.config.hsic.weight
        else:
            self.__hsic_weight = self.config.hsic.start_weight + (
                    self.trainer.current_epoch // self.config.hsic.frequency) * self.config.hsic.step

    def __step(self, batch, is_train=True):
        x, attr = batch

        y = attr['Blond_Hair']
        group_indices = attr['group_idx']
        c = one_hot(attr['Male'] % 2, num_classes=2).float()

        y_hat, emb = self.network.get_y_and_emb(x)

        hsic = HSIC(emb, c)

        cross_entropies = self.__cross_entropy(y_hat, y)

        group_cross_entropy, group_acc, group_counts = self.__get_group_metrics(y, y_hat, group_indices, cross_entropies)

        if self.config.trainer.group_dro and is_train:
            self.__group_weights = self.__group_weights * torch.exp(
                self.config.trainer.group_weight_step * group_cross_entropy.to('cpu'))
            self.__group_weights = (self.__group_weights / (self.__group_weights.sum()))

            group_indices_one_hot = one_hot(group_indices, 4).float()

            weights = torch.matmul(group_indices_one_hot, self.__group_weights.to(group_indices_one_hot.device))

            cross_entropies *= torch.tensor(weights)

        cross_entropy = cross_entropies.mean()
        acc = accuracy(y_hat, y)

        self.__update_hsic_weight()

        loss = cross_entropy + self.__hsic_weight * hsic

        metrics = {
            'acc': acc,
            'loss': loss,
            'hsic': hsic,
            'cross_entropy': cross_entropy,

            'group_cross_entropy': group_cross_entropy.to('cpu'),
            'group_acc': group_acc.to('cpu'),
            'group_counts': group_counts.to('cpu')
        }

        return metrics

    def __calculate_epoch_metrics(self, outputs: List[Any]) -> dict:
        acc = 0
        loss = 0
        hsic = 0
        cross_entropy = 0
        group_cross_entropy = torch.zeros(4)
        group_acc = torch.zeros(4)
        group_counts = torch.zeros(4)
        for o in outputs:
            group_cross_entropy += o["group_cross_entropy"] * o["group_counts"]
            group_acc += o["group_acc"] * o["group_counts"]
            group_counts += o["group_counts"]
            acc += o["acc"]
            loss += o["loss"]
            hsic += o["hsic"]
            cross_entropy += o["cross_entropy"]

        steps_count = len(outputs)
        group_cross_entropy /= group_counts
        group_acc /= group_counts
        acc /= steps_count
        loss /= steps_count
        cross_entropy /= steps_count
        hsic /= steps_count

        epoch_metrics_sep = {
            'acc': acc,
            'loss': loss,
            'hsic': hsic,
            'cross_entropy': cross_entropy,

            "cross_entropy_group_0": group_cross_entropy[0],
            "cross_entropy_group_1": group_cross_entropy[1],
            "cross_entropy_group_2": group_cross_entropy[2],
            "cross_entropy_group_3": group_cross_entropy[3],

            'acc_group_0': group_acc[0],
            'acc_group_1': group_acc[1],
            'acc_group_2': group_acc[2],
            'acc_group_3': group_acc[3],
        }

        epoch_metrics_shared = {
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],

            "hsic_weight": self.__hsic_weight,

            'w_group_0': self.__group_weights[0],
            'w_group_1': self.__group_weights[1],
            'w_group_2': self.__group_weights[2],
            'w_group_3': self.__group_weights[3],
        }

        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)

        return epoch_metrics_sep


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    log.info(f"\nGiven Config:\n {OmegaConf.to_yaml(cfg)}")

    config = ConfigParser.parse_config(cfg)

    log.info(f"\nResolved Dataclass:\n {config} \n")

    utils.fix_seeds(config.random_seed)

    logger = config.logger.get_logger(config.logs_root_dir)
    datamodule = config.dataset.get_datamodule()

    model = Model(cfg)
    model.datamodule = datamodule

    trainer = config.trainer.get_trainer(logger, config.logs_root_dir)

    trainer.fit(model)


if __name__ == '__main__':
    main()
