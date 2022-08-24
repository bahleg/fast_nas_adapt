import dataclasses
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm
from model import ResNet18

from config import ExpConfig


class Trainer:
    def __init__(self,
                 model: ResNet18,
                 config: ExpConfig,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 ):
        self.config = config
        self.model = model.to(self.config.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        os.makedirs(self.config.log_dir, exist_ok=True)
        with open(os.path.join(self.config.log_dir, 'config.json'), 'w') as fout:
            fout.write(json.dumps(dataclasses.asdict(config), indent=2))
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = Accuracy().to(config.device)

        # TODO: check state dict of the optimizers
        self.layer_wise_opt = {}
        for layer in self.model.model._modules:
            params = list(getattr(self.model.model, layer).parameters())
            if len(params) > 0:
                self.layer_wise_opt[layer] = torch.optim.SGD(params,
                                                             lr=self.config.lr, momentum=self.config.momentum)

        # for proposed strategy
        aux_input = torch.randn(1, 3, self.config.img_size, self.config.img_size).to(self.config.device)
        with torch.no_grad():
            _, interm_repr = self.model(aux_input)

        self.variational_modules = nn.ModuleDict()
        self.variational_optimizers = {}
        for i, layer in enumerate(interm_repr):
            # do not include last fc layer
            if layer == 'fc':
                continue
            hidden_size = interm_repr[layer].size(1)
            self.variational_modules.update({
                layer: nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten(),
                                     nn.Linear(hidden_size, self.config.num_classes)).to(self.config.device)})

            self.variational_optimizers[layer] = torch.optim.SGD(self.variational_modules[layer].parameters(),
                                                                 lr=self.config.lr, momentum=self.config.momentum)
        self.lam = nn.Parameter(torch.ones(len(interm_repr)).to(self.config.device), requires_grad=False)
        # TODO: add prior

        self._global_step = 0

    def training_step(self, batch):
        self.model.train()
        for layer in self.layer_wise_opt:
            self.layer_wise_opt[layer].zero_grad()
        x = batch['x']
        y = batch['y']
        logits, selected_out = self.model(x)

        if self.config.strategy == 'aggressive':
            loss = self.loss_fn(logits, y)
            loss.backward()
            for layer in self.layer_wise_opt:
                self.layer_wise_opt[layer].step()
        elif self.config.strategy == 'layer-wise':
            loss = self.loss_fn(logits, y)
            loss.backward()
            # TODO: ordered choice
            layer = np.random.choice(list(self.layer_wise_opt.keys()))
            self.layer_wise_opt[layer].step()
        elif self.config.strategy == 'proposed':
            layer = np.random.choice(list(self.variational_optimizers.keys()))
            var_logits = self.variational_modules[layer](selected_out[layer])
            loss = self.loss_fn(var_logits, y)
            loss.backward()
            if layer in self.layer_wise_opt:
                self.layer_wise_opt[layer].step()
            self.variational_optimizers[layer].step()
        else:
            raise NotImplementedError

        self.writer.add_scalar('Train/loss_step', loss.item(), self._global_step)
        acc = self.metric(logits.argmax(-1), y)
        self.writer.add_scalar('Train/acc_step', acc.item(), self._global_step)
        return {
            'loss': loss.item(),
            'acc': acc.item(),
        }

    @torch.no_grad()
    def validation_step(self, batch):
        self.model.eval()
        x = batch['x']
        y = batch['y']
        logits, selected_out = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits.argmax(-1), y)
        return {
            'loss': loss.item(),
            'acc': acc.item()
        }

    def train_one_epoch(self):
        for i, (x, y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train:'):
            self.training_step({'x': x.to(self.config.device), 'y': y.to(self.config.device)})
            self._global_step += 1

    def validate(self):
        self.metric.reset()
        losses = []
        for i, (x, y) in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='Valid:'):
            out = self.validation_step({'x': x.to(self.config.device), 'y': y.to(self.config.device)})
            losses.append(out['loss'])
        self.writer.add_scalar('Valid/acc_epoch', self.metric.compute().item(), self._global_step)
        self.writer.add_scalar('Valid/loss_epoch', sum(losses) / len(losses), self._global_step)

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            if self._has_checkpoint(epoch):
                self._load_checkpoint(epoch)
                print(f'Loaded checkpoint {epoch}')
                continue
            self.train_one_epoch()
            self.validate()
            self._save_checkpoint(epoch)

    def _has_checkpoint(self, epoch: int):
        model_path = os.path.join(self.config.log_dir, f'model_{epoch}.ckpt')
        return os.path.exists(model_path)

    def _save_checkpoint(self, epoch: int):
        # save only model
        model_path = os.path.join(self.config.log_dir, f'model_{epoch}.ckpt')
        torch.save(self.model.state_dict(), model_path)

    def _load_checkpoint(self, epoch):
        model_path = os.path.join(self.config.log_dir, f'model_{epoch}.ckpt')
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
