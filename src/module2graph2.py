from typing import Dict, Callable

import torch
import torch.fx
from torch.fx.node import Node
from functools import reduce, partial

import re 
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
  
def gamma_hook(module, inp, out, gamma,  intermediate_dict, label):
    result = out * gamma()
    intermediate_dict[label] = result 
    return result 

def make_gamma_hooks(module: torch.nn.Module, graph_module, gamma_constructor: Callable):
    nodes = set()
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            nodes.add(node.target)
    module.gammas = []
    module.intermediate = {}

    for node in nodes:
        submodule = get_module_by_name(module, node)
        new_gamma = gamma_constructor()
        node_propper_name = str(node.replace('.', '_'))
        submodule.add_module(f'gamma_{node_propper_name}', new_gamma)
        submodule.register_forward_hook(partial(gamma_hook, gamma = new_gamma, intermediate_dict = module.intermediate,
                                                 label = node_propper_name))
        for param in new_gamma.parameters():
            module.gammas.append(param)


def make_graph_description(graph_module, operations = None):
    nonwords = re.compile('[^A-Za-z0-9\s]+')
    if operations is None:
        operations = ['call_module']

    e = {}
    for node in graph_module.graph.nodes:
        if node.op in operations:
            #почему-то аргументы идут с нижним подчеркивание вместо точки
            in_ = nonwords.sub('.', node.name)
            e[(in_,  node.op)] = []
            for out_ in list(node.args) + list(node.kwargs):
                #print (str(out_), nonwords.sub(' ', str(out_)))
                try:
                    out_ = nonwords.sub('.', out_.name)
                except:
                    out_ = nonwords.sub('.', str(out_))
               
                e[(in_,  node.op)].append((out_))
    return e 

def networkx_plot_graph(graph_module, operations=None):
    try:
        import networkx as nx 
    except:
        print ('install networkx')
        return 
    if operations is None:
        operations = ['call_module']

    e = make_graph_description(graph_module, operations=operations)
    color_map = ['b', 'g', 'r']
    
    graph = nx.DiGraph()
    vert_colors = {}
    for in_ in e:
        in_, op = in_ 
        vert_colors[in_] = color_map[operations.index(op)]
        
        for out_ in e[(in_, op)]:
            graph.add_edge(out_, in_)
    vert_color_list = []
    for node in graph.nodes:
        vert_color_list.append(vert_colors.get(node, 'gray'))

    return graph, vert_color_list 