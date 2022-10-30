from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExpConfig:
    img_size: int = 33
    classes: Tuple[int] = (0, 1, 2, 3, 4, 5, 6, 7)

    # trainer config
    strategy: str = 'proposed'  # aggressive, layer-wise, infomax, proposed, last-layer
    batch_size: int = 128
    num_epochs: int = 20
    log_dir: str = 'results/proposed'
    device: str = 'cpu'

    # optimizer config
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    use_scheduler: bool = False
