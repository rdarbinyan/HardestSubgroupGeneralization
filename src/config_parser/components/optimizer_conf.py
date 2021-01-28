from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class OptimizerConf(ABC):
    name: str

    @abstractmethod
    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        pass


@dataclass(frozen=True)
class AdamConf(OptimizerConf):
    lr: float

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=model_params, **asdict_filtered(self))

@dataclass(frozen=True)
class SgdConf(OptimizerConf):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=model_params, **asdict_filtered(self))

valid_names = {
    "adam": AdamConf,
    "sgd": SgdConf
}


def get_config_obj(cfg_subgroup: DictConfig) -> OptimizerConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="optimizer"
    )
    return validated_dataclass
