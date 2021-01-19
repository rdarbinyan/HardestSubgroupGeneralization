import dataclasses
from typing import Optional, Sequence, Dict

from omegaconf import DictConfig, OmegaConf


def asdict_filtered(obj, remove_keys: Optional[Sequence[str]] = None) -> Dict:
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Not a dataclass/dataclass instance")

    if remove_keys is None:
        remove_keys = ["name"]

    args = dataclasses.asdict(obj)
    for key in remove_keys:
        if key in args:
            args.pop(key)

    return args


def get_config_obj_generic(cfg_group: DictConfig, dataclass_dict: Dict, config_category: str = "option"):
    if not OmegaConf.is_config(cfg_group):
        raise ValueError(f"Given config not an OmegaConf config. Got: {type(cfg_group)}")

    name = cfg_group.name
    if name is None:
        raise KeyError(
            f"The given config does not contain a 'name' entry. Cannot map to a dataclass.\n"
            f"  Config:\n {OmegaConf.to_yaml(cfg_group)}"
        )

    cfg_asdict = OmegaConf.to_container(cfg_group, resolve=True)

    try:
        dataclass_obj = dataclass_dict[name](**cfg_asdict)
    except KeyError:
        raise ValueError(
            f"Invalid Config: '{cfg_group.name}' is not a valid {config_category}. "
            f"Valid Options: {list(dataclass_dict.keys())}"
        )

    return dataclass_obj
