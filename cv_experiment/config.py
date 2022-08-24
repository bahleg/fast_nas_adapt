from dataclasses import dataclass
from typing_extensions import Literal


@dataclass
class ExpConfig:
    img_size: int = 229
    num_classes: int = 10

    # trainer config
    strategy: Literal['aggressive', 'layer-wise', 'proposed'] = 'proposed'
    batch_size: int = 32
    num_epochs: int = 20
    log_dir: str = 'results/proposed'
    device: str = 'cuda'

    # optimizer config
    lr: float = 1e-3
    momentum: float = 0.9
