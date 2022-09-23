from dataclasses import dataclass


@dataclass
class ExpConfig:
    img_size: int = 229
    num_classes: int = 10

    # trainer config
    strategy: str = 'proposed'  # aggressive, layer-wise, infomax, proposed, last-layer
    batch_size: int = 32
    num_epochs: int = 20
    log_dir: str = 'results/proposed'
    device: str = 'cuda'

    # optimizer config
    lr: float = 1e-3
    momentum: float = 0.9
