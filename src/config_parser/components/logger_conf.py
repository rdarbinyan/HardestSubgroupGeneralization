import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers.base import LightningLoggerBase

from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class LoggerConf(ABC):
    name: str

    @abstractmethod
    def get_logger(self, *args) -> Optional[LightningLoggerBase]:
        pass


@dataclass(frozen=True)
class DisabledLoggerConf(LoggerConf):
    @staticmethod
    def get_logger(*args) -> None:
        return False


@dataclass(frozen=True)
class TensorboardConf(LoggerConf):
    run_name: str
    run_version: Optional[Union[int, str]]

    def get_logger(self, save_dir: Path, *args) -> LightningLoggerBase:
        args_dict = asdict_filtered(self)
        run_name = args_dict.pop("run_name")
        run_version = args_dict.pop("run_version")

        tb_logger = pl_loggers.TensorBoardLogger(name=run_name, version=run_version, save_dir=str(save_dir), **args_dict)

        return tb_logger


valid_names = {
    "disabled": DisabledLoggerConf,
    "tensorboard": TensorboardConf
}


def get_config_obj(cfg_subgroup: DictConfig) -> LoggerConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="logger"
    )
    return validated_dataclass
