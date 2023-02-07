from typing import Dict

import torch
import torch.fx
from torch.fx.node import Node

import numpy as np 

class GraphInterperterSimple:
    """
    see torch.fx shape propagation
    """

    def __init__(self, mod):
        try:
            mod.graph
        except:
            mod = torch.fx.symbolic_trace(mod)
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def __call__(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args),
                                     **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs))
            if node.op == 'output':
                return result

            env[node.name] = result


class GraphInterperterWithGamma(torch.nn.Module):
    """
    see torch.fx shape propagation
    """
    def __init__(self, mod):
        super(GraphInterperterWithGamma, self).__init__()
        try:
            mod.graph
        except:
            mod = torch.fx.symbolic_trace(mod)
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.init_gammas()

    def init_gammas(self):
        i = 0
        gammas = []
        self.gammas_name = {}
        for node in self.graph.nodes:
            if node.op == 'call_module':
                gammas.append(1.0)
                self.gammas_name[str(node)] = i# перевод в str тут для удобства. в реалньых методах это не нужно
                i+=1                        # да и вообще, тут по идее должен быть тензор/параметр
        self.gammas =  torch.nn.Parameter(torch.as_tensor(gammas), requires_grad = False)

    def sample_gammas(self):
        return self.gammas 

    def make_gammas_discrete(self):
        raise NotImplementedError()

    def forward(self, *args, intermediate=False):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        sampled_gammas = self.sample_gammas()
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs)) * sampled_gammas[self.gammas_name[str(node)]]
            if node.op == 'output':
                if intermediate:
            
                    return result, env 
        
                return result
                        
            env[node.name] = result
        
        return result
        
class GraphInterperterWithBernGamma(GraphInterperterWithGamma):
    def __init__(self, mod, gamma_shift=0.0):
        self.gamma_shift = gamma_shift
        super().__init__(mod)

    def init_gammas(self):
        i = 0
        gammas = []
        self.gammas_name = {}
        for node in self.graph.nodes:
            if node.op == 'call_module':
                gammas.append(np.random.randn()+self.gamma_shift)
                self.gammas_name[str(node)] = i# перевод в str тут для удобства. в реалньых методах это не нужно
                i+=1                        # да и вообще, тут по идее должен быть тензор/параметр
        self.gammas =  torch.nn.Parameter(torch.as_tensor(gammas), requires_grad = True)
        self.discrete = False 

    def sample_gammas(self):
        if self.discrete:
            return self.gammas
        else:
            return torch.sigmoid(self.gammas)
        
    def make_gammas_discrete(self):
        self.gammas.data = (self.gammas.data>=0) * 1.0
        self.gammas.requires_grad = False 
        self.discrete = True
