from typing import Optional, Any

from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.config_parser.components import (
    dataset_conf,
    optimizer_conf,
    network_conf,
    scheduler_conf,
    logger_conf,
    trainer_conf,

    DatasetConf,
    OptimizerConf,
    NetworkConf,
    SchedulerConf,
    LoggerConf,
    TrainerConf,
)


@dataclass(frozen=True)
class TrainConf:
    random_seed: Optional[int]
    logs_root_dir: str
    dataset: DatasetConf
    optimizer: OptimizerConf
    network: NetworkConf
    scheduler: SchedulerConf
    logger: LoggerConf
    trainer: TrainerConf
    hsic: Any  # DictConfig (pydantic gives error on DictConfig)


class ConfigParser:
    @classmethod
    def parse_config(cls, cfg: DictConfig):
        config = TrainConf(
            random_seed=cfg.random_seed,
            logs_root_dir=cfg.logs_root_dir,
            dataset=dataset_conf.get_config_obj(cfg.dataset),
            optimizer=optimizer_conf.get_config_obj(cfg.optimizer),
            network=network_conf.get_config_obj(cfg.network),
            scheduler=scheduler_conf.get_config_obj(cfg.scheduler),
            logger=logger_conf.get_config_obj(cfg.logger),
            trainer=trainer_conf.get_config_obj(cfg.trainer),
            hsic=cfg.hsic
        )
        return config
