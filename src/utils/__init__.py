from typing import Optional

import pytorch_lightning as pl


def fix_seeds(random_seed: Optional[int]) -> None:
    if random_seed is not None:
        pl.seed_everything(random_seed)
