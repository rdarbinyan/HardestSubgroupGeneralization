from abc import ABC, abstractmethod

import pytorch_lightning as pl
from pydantic.dataclasses import dataclass
from pytorch_lightning.loggers.base import LightningLoggerBase
from omegaconf import DictConfig

from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class TrainerConf(ABC):
    name: str

    @abstractmethod
    def get_trainer(self, pl_logger: LightningLoggerBase, default_root_dir: str) -> pl.Trainer:
        pass


@dataclass(frozen=True)
class StandardTrainerConfig(TrainerConf):
    gpus: int
    precision: int
    max_epochs: int
    checkpoint_callback: bool # in the future, will checkpoint type will be allowed
    group_dro: bool
    group_weight_step: float

    log_every_n_steps: int
    progress_bar_refresh_rate: int

    def get_trainer(self, pl_logger: LightningLoggerBase, default_root_dir: str) -> pl.Trainer:
        args = asdict_filtered(self)

        del args["group_dro"]
        del args["group_weight_step"]

        trainer = pl.Trainer(
            logger=pl_logger,
            default_root_dir=default_root_dir,
            **args
        )

        return trainer


valid_names = {"standard": StandardTrainerConfig}


def get_config_obj(cfg_subgroup: DictConfig) -> TrainerConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="trainer"
    )
    return validated_dataclass
