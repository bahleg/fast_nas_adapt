# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN, SigmoidNode, GSNode
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy
from nni.retiarii import fixed_arch
import types
import torch 

logger = logging.getLogger('nni')

def monkey_patch_parameters(self):
    for name, param in self.named_parameters():
        if 'gamma' not in name:    
            yield param

def monkey_patch_parameters_head(self):
    for param in self.linear.parameters():
            yield param
            
                        
if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--gs", default=False, action="store_true")    
    parser.add_argument("--arc-checkpoint", default="./checkpoint.json")
    parser.add_argument("--pt", default="./darts.pt")
    parser.add_argument("--out", default='out.pt')    
    parser.add_argument("--head", default=False, action="store_true")        
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("fmnist3")

    with fixed_arch(args.arc_checkpoint):
        if args.gs:
            node_cls = GSNode
        else:
            node_cls = SigmoidNode
            
        model = CNN(28, 1, 16, 8, args.layers, auxiliary=False, node_cls = node_cls)
        model.load_state_dict(torch.load(args.pt), strict=False)
        model.aux_pos = -1
        model.linear = nn.Linear(model.linear.weight.shape[1], 2) # weight is class * features matrix
        if not args.head:
            model.parameters = types.MethodType(monkey_patch_parameters, model) 
        else:
            model.parameters = types.MethodType(monkey_patch_parameters_head, model)  # only head parameters
            
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    from darts_adaptive import DartsSigmaTrainer
    trainer = DartsSigmaTrainer(
        model=model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=dataset_train,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled
    )
    trainer.fit()
    final_architecture = trainer.export()
    torch.save(model.state_dict(), args.out)

