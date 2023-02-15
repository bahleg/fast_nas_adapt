import sys
sys.path.append('../src/')
import importlib
from matplotlib import pylab as plt

import numpy as np
import torch
import torchvision

import cifar_data
import resnet18
import module2graph
import utils
import dartslike

import pickle
import os
import gc
import argparse

class LowRankLinear(torch.nn.Module):
    def __init__(self, in_, out_, dim=1):
        super().__init__()
        self.l = torch.nn.Parameter(torch.randn(in_, dim)*1e-3)
        self.r = torch.nn.Parameter(torch.randn(dim, out_) * 1e-3)
    
    def forward(self, x):
        #print (x.shape, self.l.shape, self.r.shape)
        return x@self.l@self.r


class Aux(torch.nn.Module):
    def __init__(self, sizes, layer_names):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_names = layer_names
        self.means_int = torch.nn.ModuleDict()
        self.lsigmas_int = torch.nn.ParameterDict()
        self.means_y = torch.nn.ModuleDict()
        self.lsigmas_y = torch.nn.ParameterDict()
        
        for i in range(len(layer_names)-1):
            current = layer_names[i]
            next_ = layer_names[i+1]
            mat_size = np.prod(sizes[current][1:]) * np.prod(sizes[next_][1:])
            if mat_size > 1024 * 1024:
                linear = LowRankLinear(np.prod(sizes[current][1:]), np.prod(sizes[next_][1:]))
            else:
                linear = torch.nn.Linear(np.prod(sizes[current][1:]), np.prod(sizes[next_][1:]))
            
            lsigma = torch.tensor(-2.0)
            self.means_int.update({current: linear})
            self.lsigmas_int.update({current: lsigma})
            self.layers.append(linear)
            
        for i in range(len(layer_names)):
            current = layer_names[i]
            linear = torch.nn.Linear(np.prod(sizes[current][1:]), 2)
            lsigma = torch.nn.Parameter(torch.tensor(-2.0))
            self.means_y.update({current: linear})
            self.lsigmas_y.update({current: lsigma})
            self.layers.append(linear)
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_opt_type", type=str)
    parser.add_argument("--gamma_opt_type", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--layer_wise", type=bool, default=False)
    args = parser.parse_args()

    parameter_opt_type, gamma_opt_type, seed, lam = args.param_opt_type, args.gamma_opt_type, args.seed, args.lam
    batch_size = 64
    device = 'cpu'
    trial_num = 1
    epoch_num = 1
    lr = 1e-3
    wd = 1e-6
    batch_seen = 5
    trainloader, valloader, testloader = cifar_data.get_dataloaders([8,9], batch_size=batch_size, need_val=True)
    model = resnet18.ResNet18(8).to(device)
    model.load_state_dict(torch.load('../data/model_last.ckpt', map_location=device))
    model.model.fc = torch.nn.Linear(512, 2).to(device)
    model = module2graph.GraphInterperterWithBernGamma(model, 1.0).to(device)
    layer_names = list(model.forward(torch.randn(64, 3, 33, 33).to(device), intermediate=True)[1].keys())
    sizes = {}
    for k,v in model(torch.randn(64, 3, 33, 33).to(device), intermediate=True)[1].items():
        sizes[k] = v.shape
    layer_names = [k for k in layer_names if k not in ['flatten', 'x']]

    name = f'{parameter_opt_type}.{gamma_opt_type}.{lam}.{seed}'
    print (name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = resnet18.ResNet18(8).to(device)
    aux = Aux(sizes, layer_names).to(device)
    model.load_state_dict(torch.load('../data/model_last.ckpt', map_location=device))
    model.model.fc = torch.nn.Linear(512, 2).to(device)
    model = module2graph.GraphInterperterWithBernGamma(model, 1.0).to(device)
    importlib.reload(dartslike)
    dl = dartslike.DartsLikeTrainer(model, parameter_optimization=parameter_opt_type, gamma_optimization=gamma_opt_type,
                                    aux=aux,MI_Y_lambda=lam, layer_wise=args.layer_wise)
    history = dl.train_loop(trainloader, valloader, testloader, batch_seen, epoch_num, lr, lr, device, wd)

    with open(name+'.pckl','wb') as out:
        out.write(pickle.dumps( (history, model.state_dict())))
