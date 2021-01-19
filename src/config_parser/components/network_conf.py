from abc import ABC, abstractmethod

import torch.nn as nn
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.networks import Resnet50
from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class NetworkConf(ABC):
    name: str

    @abstractmethod
    def get_network(self) -> nn.Module:
        pass


@dataclass(frozen=True)
class Resnet50Conf(NetworkConf):
    pretrained: bool
    num_target_classes: int

    def get_network(self) -> nn.Module:
        return Resnet50(**asdict_filtered(self))


valid_names = {"resnet50": Resnet50Conf}


def get_config_obj(cfg_subgroup: DictConfig) -> NetworkConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="network"
    )
    return validated_dataclass
