import dataclasses
import json
import os
from pydoc import stripid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm
from model import ResNet18

from config import ExpConfig


class IdTransform(nn.Module):
    def __init__(self):
        super(IdTransform, self).__init__()
        self.unused_param = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor):
        return x


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
        self.layer_wise_sch = {}
        for layer in self.model.model._modules:
            params = list(getattr(self.model.model, layer).parameters())
            if len(params) > 0:
                self.layer_wise_opt[layer] = torch.optim.SGD(params,
                                                             lr=self.config.lr, momentum=self.config.momentum,
                                                             weight_decay=self.config.weight_decay)
                self.layer_wise_sch[layer] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.layer_wise_opt[layer], T_max=self.config.num_epochs)

        aux_input = torch.randn(1, 3, self.config.img_size, self.config.img_size).to(self.config.device)
        with torch.no_grad():
            _, interm_repr = self.model(aux_input)

        # INFO-MAX: p(y|v)
        self.variational_modules_infomax = nn.ModuleDict()
        self.variational_opt_infomax = {}
        for i, layer in enumerate(interm_repr):
            if layer == 'fc':
                self.variational_modules_infomax.update({
                    'fc': IdTransform(),
                })
            else:
                hidden_size = interm_repr[layer].size(1)
                self.variational_modules_infomax.update({
                    layer: nn.Sequential(nn.AdaptiveAvgPool2d((8, 8)),
                                         nn.Flatten(),
                                         nn.Linear(hidden_size * 8**2, len(self.config.classes))).to(self.config.device)})

            self.variational_opt_infomax[layer] = torch.optim.SGD(self.variational_modules_infomax[layer].parameters(),
                                                                 lr=self.config.lr, momentum=self.config.momentum)
        self.lam = nn.Parameter(torch.ones(len(interm_repr)).to(self.config.device), requires_grad=False)
        # TODO: add prior

        # The proposed: p(v_{i+1}|v_i), p(y|v_last)
        self.variational_modules_proposed = nn.ModuleDict()
        self.variational_opt_proposed = {}
        # TODO: automatically init var_modules
        self.variational_modules_proposed.update({
            'conv1': nn.Conv2d(3, 64, padding=1, kernel_size=3, stride=2)
        })
        for key in ['bn1', 'relu']:
            self.variational_modules_proposed.update({
                key: nn.Conv2d(64, 64, kernel_size=3, padding=1)
            })
        self.variational_modules_proposed.update({
            'maxpool': nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        })
        cur_channels = 64
        for layer_id in range(1, 5):
            stride = 2
            if layer_id == 1:
                stride = 1
                out_channels = cur_channels
            else:
                out_channels = 2 * cur_channels
            self.variational_modules_proposed.update({
                f'layer{layer_id}': nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            })
            if layer_id != 1:
                cur_channels *= 2
        self.variational_modules_proposed.update({
            'avgpool': nn.AdaptiveAvgPool2d(output_size=(1, 1))
        })
        # p(y|v_last)
        self.variational_modules_proposed.update({
            'fc': nn.Sequential(nn.Flatten(), nn.Linear(512, len(self.config.classes)))
        })
        # optimizers
        for layer in self.variational_modules_proposed:
            params = list(self.variational_modules_proposed[layer].parameters())
            if len(params) > 0:
                self.variational_opt_proposed[layer] = torch.optim.SGD(params, lr=self.config.lr, momentum=self.config.momentum)
        self.variational_modules_proposed.to(self.config.device)
        
        # TODO: add sigma^2 for each layer
        self.mse_loss = nn.MSELoss()
        self.mse_weight = torch.tensor(self.config.mse_weight).to(self.config.device)

        self._global_step = 0

    def training_step(self, batch):
        self.model.train()
        for layer in self.layer_wise_opt:
            self.layer_wise_opt[layer].zero_grad()
        for layer in self.variational_opt_infomax:
            self.variational_opt_infomax[layer].zero_grad()
        for layer in self.variational_opt_proposed:
            self.variational_opt_proposed[layer].zero_grad()
        x = batch['x']
        y = batch['y']
        logits, selected_out = self.model(x)

        if self.config.strategy == 'aggressive':
            loss = self.loss_fn(logits, y)
            loss.backward()
            for layer in self.layer_wise_opt:
                self.layer_wise_opt[layer].step()
                self.layer_wise_sch[layer].step()
        elif self.config.strategy == 'last-layer':
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.layer_wise_opt['fc'].step()
        elif self.config.strategy == 'layer-wise':
            loss = self.loss_fn(logits, y)
            loss.backward()
            # TODO: ordered choice
            layer = np.random.choice(list(self.layer_wise_opt.keys()))
            self.layer_wise_opt[layer].step()
        elif self.config.strategy == 'infomax':
            layer = np.random.choice(list(self.variational_opt_infomax.keys()))
            var_logits = self.variational_modules_infomax[layer](selected_out[layer])
            loss = self.loss_fn(var_logits, y)
            loss.backward()
            if layer in self.layer_wise_opt:
                self.layer_wise_opt[layer].step()
            self.variational_opt_infomax[layer].step()
        elif self.config.strategy == 'proposed':
            # next layer
            next_layer_idx  = np.random.choice(len(list(self.variational_opt_proposed.keys())))
            next_layer = list(self.variational_modules_proposed.keys())[next_layer_idx]
            cur_layer = list(self.variational_modules_proposed.keys())[next_layer_idx - 1]
            # print(cur_layer, next_layer, self.variational_modules_proposed)
            if next_layer_idx == 0:
                var_logits = self.variational_modules_proposed['conv1'](batch['x'])
            else:
                var_logits = self.variational_modules_proposed[next_layer](selected_out[cur_layer])
            if next_layer == 'fc':
                loss = self.loss_fn(var_logits, batch['y']) + \
                    self.mse_loss(var_logits, selected_out['fc']) * self.mse_weight
            else:
                loss = self.mse_loss(var_logits, selected_out[next_layer]) * self.mse_weight
            loss.backward()
            if next_layer in self.variational_opt_proposed:
                self.variational_opt_proposed[next_layer].step()
            if next_layer in self.layer_wise_opt:
                self.layer_wise_opt[next_layer].step()
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
        for i, (x, y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train', leave=False):
            self.training_step({'x': x.to(self.config.device), 'y': y.to(self.config.device)})
            self._global_step += 1

    def validate(self):
        self.metric.reset()
        losses = []
        for i, (x, y) in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='Valid', leave=False):
            out = self.validation_step({'x': x.to(self.config.device), 'y': y.to(self.config.device)})
            losses.append(out['loss'])
        self.writer.add_scalar('Valid/acc_epoch', self.metric.compute().item(), self._global_step)
        self.writer.add_scalar('Valid/loss_epoch', sum(losses) / len(losses), self._global_step)

    def fit(self, num_epochs: int):
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

    def _load_checkpoint(self, epoch: int):
        model_path = os.path.join(self.config.log_dir, f'model_{epoch}.ckpt')
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
