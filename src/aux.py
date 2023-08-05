import torch 
import numpy as np

class LowRankLinear(torch.nn.Module):
    def __init__(self, in_, out_, dim=1):
        super().__init__()
        self.l = torch.nn.Parameter(torch.randn(in_, dim)*1e-3)
        self.r = torch.nn.Parameter(torch.randn(dim, out_) * 1e-3)
    
    def forward(self, x):
        #print (x.shape, self.l.shape, self.r.shape)
        return x@self.l@self.r


class Aux(torch.nn.Module):
    def __init__(self, sizes, layer_names, node2node, class_num=2, simple:bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_names = layer_names
        if not simple:
            self.means_int = torch.nn.ModuleDict()
            self.lsigmas_int = torch.nn.ParameterDict()
            self.lsigmas_y = torch.nn.ParameterDict()
        self.means_y = torch.nn.ModuleDict()
        self.class_num = class_num
        sizes['output'] = [0, class_num] # sizes are [batch size, feature num]. 

        if not simple:
            for current in node2node:
                for next_ in node2node[current]:
                  
                    mat_size = np.prod(sizes[current][1:]) * np.prod(sizes[next_][1:])
                    if mat_size > 1024 * 1024:
                        linear = LowRankLinear(np.prod(sizes[current][1:]), np.prod(sizes[next_][1:]))
                    else:
                        print (np.prod(sizes[current][1:]), next_, sizes[next_], np.prod(sizes[next_][1:])) 
                        linear = torch.nn.Linear(np.prod(sizes[current][1:]), np.prod(sizes[next_][1:]))

                    lsigma = torch.tensor(-2.0)
                    self.means_int.update({current+'_____'+ next_: linear})
                    self.lsigmas_int.update({current + '_____'+ next_: lsigma})
                    self.layers.append(linear)

        for i in range(len(layer_names)):
            current = layer_names[i]
            linear = torch.nn.Linear(np.prod(sizes[current][1:]), self.class_num)
            self.means_y.update({current: linear})
            self.layers.append(linear)

            if not simple:
                lsigma = torch.nn.Parameter(torch.tensor(-2.0))
                self.lsigmas_y.update({current: lsigma})
            
