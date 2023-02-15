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


@torch.no_grad()
def safe_prune(model, trainloader, num_to_prune, gammas_to_prune):
    for x,y in trainloader:
        break
    num = 0
    for g in gammas_to_prune:
            model.gammas[g] = 1.0
    
    i = 0
    while num < num_to_prune and i < len(gammas_to_prune):
        model.gammas[i] = 0.0
  
        if abs(model(x) - model(torch.zeros(x.shape))).sum() < 1e-5:
            model.gammas[i] = 1.0
            i += 1
        else:
            num += 1
            i += 1
    if num == num_to_prune:
        return True
    else:
        return False 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_opt_type", type=str)
    parser.add_argument("--gamma_opt_type", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lam", type=float)
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
    model.model.fc = torch.nn.Linear(512, 2).to(device)
    model = module2graph.GraphInterperterWithBernGamma(model, 1.0).to(device)
    name = f'{parameter_opt_type}.{gamma_opt_type}.{lam}.{seed}'
    with open(name+'.pckl', 'rb') as inp:
        _, state_dict = pickle.loads(inp.read())

    model.load_state_dict(state_dict)
    model.eval()
    model.gammas.requires_grad = False 
    model.discrete = True
    argsort_gammas = np.argsort(model.gammas.data.numpy())
    results = []
    print(f'Gammas: {len(model.gammas)}')

    # l_num, r_num = 0, len(model.gammas) - 1
    # while l_num < r_num - 1:
    #     num_to_prune = (l_num + r_num) // 2
    #     model.load_state_dict(state_dict)
    #     if safe_prune(model, trainloader, num_to_prune, argsort_gammas):
    #         print(num_to_prune)
    #         results.append(utils.train_loop(model, trainloader, testloader, batch_seen, epoch_num, lr, device)[-1])
    #         l_num = num_to_prune
    #     else:
    #         print(num_to_prune, 'failed')
    #         r_num = num_to_prune
    for num_to_prune in (range(len(model.gammas))):

        model.load_state_dict(state_dict)
        if safe_prune(model, trainloader, num_to_prune, argsort_gammas):
            print (num_to_prune)
            results.append(utils.train_loop(model, trainloader, testloader, batch_seen, epoch_num, lr, device)[-1])
        else:
            print (num_to_prune, 'failed')
            break

    with open(name+'tuned.pckl','wb') as out:
        out.write(pickle.dumps( (results, model.state_dict())))

