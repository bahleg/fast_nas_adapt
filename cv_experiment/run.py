from model import ResNet18
from data import get_dataloaders
from config import ExpConfig
from simple_parsing import ArgumentParser
from trainer import Trainer
import torch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(ExpConfig(), dest='config')
    args = parser.parse_args()
    config: ExpConfig = args.config
    print(config)

    model = ResNet18(num_classes=len(config.classes))
    model_state_dict = torch.load(config.model_path, map_location='cpu')
    model_state_dict = {key: model_state_dict[key] for key in model_state_dict if 'fc' not in key}
    model.model.load_state_dict(model_state_dict, strict=False)
    train_dl, test_dl = get_dataloaders(classes=config.classes, batch_size=config.batch_size,
                                        img_size=config.img_size)

    trainer = Trainer(model, config, train_loader=train_dl, valid_loader=test_dl)
    trainer.fit(config.num_epochs)

