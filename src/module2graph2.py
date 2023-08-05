from typing import Dict, Callable

import torch
import torch.fx
from torch.fx.node import Node
from functools import reduce, partial
class Gamma(torch.nn.Module):
    def __init__(self, initial_gamma = 0.0) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(initial_gamma))
        self.discrete = False 
    
    def make_discrete(self):
        raise NotImplementedError()
    

    def forward(self):
        return self.param  
    
class SigmoidGamma(Gamma):
    def make_discrete(self):
        self.discrete = True 
        self.param.requires_grad = False 

    def forward(self):
        if self.discrete:
            return 1.0 * (self.param)
        else:
            return torch.sigmoid(self.param)

        
def get_module_by_name(module,
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
  
def gamma_hook(module, inp, out, gamma,  intermediate_dict, proper2real_label, real2proper_label, label, proper_label):
    result = out * gamma()
    intermediate_dict[proper_label] = result 
    proper2real_label[proper_label] = label 
    real2proper_label[label] = proper_label

    return result 

def make_gamma_hooks(module: torch.nn.Module, graph_module, gamma_constructor: Callable):
    node2node = {}
    for node in graph_module.graph.nodes:
      for node2 in node.all_input_nodes:
        if node2 not in node2node:
          node2node[node2] = []
        node2node[node2].append(node)


    modified = True
    while modified:
      modified = False 
      for node in node2node:
        to_del = set()
        for i in range(len(node2node[node])):
          if node2node[node][i].op != 'call_module':
            modified = True
            to_del.add(node2node[node][i])
            if node2node[node][i] in node2node: # check that not output
              node2node[node].extend(node2node[node2node[node][i]])
          for node2 in to_del:
            node2node[node].remove(node2)
            
    to_del = set()
    to_change = set()

    for node in node2node:
      if node.op != 'call_module':
        to_del.add(node)
      else:
        to_change.add(node)
    for node in to_del:
      del node2node[node]  
    for node in to_change:
      new_result = [str(node2.target).replace('.', '_') for node2 in node2node[node]]
      node2node[str(node.target).replace('.', '_')] = new_result
      del node2node[node]
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            tgt = node.target.replace('.', '_')
            if (tgt not in node2node) or (len(node2node[tgt]) == 0):
          
                print (str(node), 'output')
                node2node[str(node)] = ['output']
    module.node2node = node2node
    
    nodes = set()

    
    for node in graph_module.graph.nodes:
        #print (node.target, node.op)
        if node.op == "call_module":
            nodes.add(node.target)
    
    module.gammas = []
    module.intermediate = {}
    module.proper2real_label = {}
    module.real2proper_label = {}

    for node in nodes:
        submodule = get_module_by_name(module, node)
        new_gamma = gamma_constructor()
        node_intermediate_name = str(node)
        node_propper_name = str(node_intermediate_name.replace('.', '_'))
        
        submodule.add_module(f'gamma_{node_propper_name}', new_gamma)
        submodule.register_forward_hook(partial(gamma_hook, gamma = new_gamma, intermediate_dict = module.intermediate,
                                                label = node_intermediate_name, 
                                                proper2real_label = module.proper2real_label,
                                                  real2proper_label = module.real2proper_label,
                                                    proper_label = node_propper_name))
        for param in new_gamma.parameters():
            module.gammas.append(param)

            
