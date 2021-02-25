from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.datasets import CelebADataModule
from src.config_parser.utils import asdict_filtered, get_config_obj_generic


@dataclass(frozen=True)
class DatasetConf(ABC):
    name: str

    @abstractmethod
    def get_datamodule(self) -> pl.LightningDataModule:
        pass


@dataclass(frozen=True)
class CelebAConf(DatasetConf):
    data_root: str
    download: bool
    batch_size: int
    num_workers: int
    confounder_name: str
    target_name: str
    sampler: Optional[str]

    def get_datamodule(self) -> pl.LightningDataModule:
        return CelebADataModule(**asdict_filtered(self))


valid_names = {"celeba": CelebAConf}


def get_config_obj(cfg_subgroup: DictConfig) -> DatasetConf:
    validated_dataclass = get_config_obj_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="dataset"
    )
    return validated_dataclass
