from abc import ABC, abstractmethod
from typing import Optional

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from torch.optim.optimizer import Optimizer

from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class SchedulerConf(ABC):
    name: str

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        pass


@dataclass(frozen=True)
class DisabledConfig(SchedulerConf):
    def get_scheduler(self, optimizer: Optimizer) -> None:
        return None


@dataclass(frozen=True)
class PlateauConfig(SchedulerConf):
    factor: float
    patience: int
    mode: str
    threshold: float
    cooldown: int
    eps: float
    verbose: bool

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **asdict_filtered(self))


valid_names = {
    "disabled": DisabledConfig,
    "plateau": PlateauConfig,
}


def get_config_obj(cfg_subgroup: DictConfig) -> SchedulerConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="scheduler"
    )
    return validated_dataclass
