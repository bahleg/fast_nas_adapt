import torch
import numpy as np
import torch
from torchmetrics import Accuracy
import tqdm
from torch.nn.functional import one_hot
from copy import deepcopy


class MILoss(torch.nn.Module):
    def __init__(self, aux, model,  MI_Y_lambda = 0.0, num_classes=2, layer_wise: bool = False) -> None:
        super().__init__()
        self.model = model
        self.aux = aux 
        self.MI_Y_lambda = MI_Y_lambda
        self.num_classes = num_classes
        self.layer_wise = layer_wise  # switch to layer wise optimization
        self.layer_id = None 
        self.layer_id2 = None
        self.mdls = dict(self.model.named_modules())
        self.opt_cnt = 0
        
        
    def select_layer(self):
        self.opt_cnt += 1
        #for p in self.model.parameters():
        #    p.requires_grad_(False)
        
        for m in self.aux.means_y.values():
            m.requires_grad_(False)
        for m in self.aux.means_int.values():
            m.requires_grad_(False)
        for m in self.aux.lsigmas_int.values():
            m.requires_grad_(False)
        for m in self.aux.lsigmas_y.values():
            m.requires_grad_(False)

        for m in self.mdls.values():
            m.requires_grad_(False)
        
        for g in self.model.gammas:
            g.requires_grad_(True)
            
        
            
        if self.opt_cnt % 2== 0:
          heads = [n for n in self.model.node2node if 'output' in self.model.node2node[n]]
          assert len(heads)==1
          self.layer_id = heads[0]
          self.layer_id2 = 'output'
        else:
          self.layer_id = np.random.choice(list(self.model.node2node.keys()))
          self.layer_id2 = np.random.choice(self.model.node2node[self.layer_id])
          
        #self.layer_id = list(self.model.real2proper_label.keys())[-1]
        #print (self.layer_id)
        
        self.mdls[self.model.proper2real_label[self.layer_id]].requires_grad_(True)
        self.aux.means_y[self.layer_id].requires_grad_(True)
        self.aux.means_int[self.layer_id+'_____'+self.layer_id2].requires_grad_(True)
        self.aux.lsigmas_int[self.layer_id+'_____'+self.layer_id2].requires_grad_(True)
        self.aux.lsigmas_y[self.layer_id].requires_grad_(True)
        
    def forward(self, out, intermediate, target):
        ### intermediate
        if self.layer_wise:
            layers = [self.layer_id]
        else:
            layers = list(self.model.node2node.keys())
        
        loss = 0.0
        target = one_hot(target, self.num_classes)
        for layer in layers:
            
            in_interm = intermediate[layer].view(intermediate[layer].shape[0], -1)
            
            
            if not self.layer_wise:
                layers2 = self.model.node2node[layer]
            else:
                layers2 = [self.layer_id2]
           
            for layer2 in layers2:
                if layer2 == 'output':
                    target_interm = target
                else:
                    target_interm = intermediate[layer2].view(intermediate[layer2].shape[0], -1)
            
            target_interm = target_interm.detach()
            mean = self.aux.means_int[layer + '_____'+ layer2](in_interm)
            log_sigma = self.aux.lsigmas_int[layer + '_____'+ layer2]
             
                          
            loss += (log_sigma * np.prod(mean.shape) + \
                (((mean - target_interm.view(target_interm.shape[0], -1))**2)
                / (2 * torch.exp(log_sigma) ** 2))).sum() \
                    * (1.0-self.MI_Y_lambda)
                

         
        
            if 'output' in self.model.node2node[layer]:
                mean = in_interm 
                log_sigma = torch.tensor(0.0).to(in_interm.device)
            else:
                mean = self.aux.means_y[layer](in_interm)
                log_sigma = self.aux.lsigmas_y[layer]
            #log_sigma = log_sigma.detach().view(1)  # detach sigma
            loss2 = 0.0
            loss2 += (log_sigma * mean.numel())
            loss2 += (((mean - target) ** 2) / (2 * torch.exp(log_sigma) ** 2)).sum()
            loss += self.MI_Y_lambda * loss2
           
        return loss 

class CELoss(torch.nn.Module):
    def __init__(self, model, aux, layer_wise: bool = False) -> None:
        super().__init__()
        self.model = model 
        self.layer_id = None 
        self.mdls = dict(self.model.named_modules())
        self.layer_wise = layer_wise
        self.aux = aux 
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt_cnt = 0

    def select_layer(self):
        self.opt_cnt += 1
        #for p in self.model.parameters():
        #    p.requires_grad_(False)
        
        for m in self.aux.means_y.values():
            m.requires_grad_(False)

        for m in self.mdls.values():
            m.requires_grad_(False)
        
        for g in self.model.gammas:
            g.requires_grad_(True)
        if self.opt_cnt % 2== 0:
          self.layer_id = list(self.model.real2proper_label.keys())[-1]
        else:
          self.layer_id = np.random.choice(list(self.model.real2proper_label.keys()))
        #self.layer_id = list(self.model.real2proper_label.keys())[-1]
        #print (self.layer_id)
        self.mdls[self.layer_id].requires_grad_(True)
        self.aux.means_y[self.model.real2proper_label[self.layer_id]].requires_grad_(True)
       

    def forward(self, out, intermediate, target):
        if not self.layer_wise:
            return self.loss_fn(out, target)
        
        required_intermediate = intermediate[self.model.real2proper_label[self.layer_id]]
        to_transform =  required_intermediate.view(required_intermediate.shape[0], -1)
        if self.layer_id == list(self.mdls.keys())[-1]:
            mean = to_transform 
        else:
            mean = self.aux.means_y[self.model.real2proper_label[self.layer_id]](to_transform)
        
        return self.loss_fn(mean, target) 


class DartsLikeTrainer:
    def __init__(self, graph_model, unrolled=False, parameter_optimization='CE', gamma_optimization='CE',
                 aux=None, MI_Y_lambda = 0.0, layer_wise: bool = False) -> None:
        self.graph_model = graph_model
        self.unrolled = unrolled
        self.parameter_optimization = parameter_optimization
        self.gamma_optimization = gamma_optimization
        self.aux = aux 
        self.MI_Y_LAMBDA = MI_Y_lambda
        self.layer_wise = layer_wise

        
        

    def train_loop(self, traindata,  valdata, testdata, sample_mod, epoch_num, lr, lr2, device, wd, intermediate_getter = None, class_num=2):
        gammas =  self.graph_model.gammas
        parameters =  [p for n,p in self.graph_model.named_parameters() if n !='gammas']
        gammas.extend(list(self.aux.parameters()))
        parameters.extend(list(self.aux.parameters()))
        
        if not self.unrolled:
            optim = torch.optim.Adam(parameters, lr=lr)
            optim2 = torch.optim.Adam(gammas, lr=lr2, weight_decay=wd)
        else:
            raise NotImplementedError("unrolled")

        history = []
        acc = Accuracy(task='multiclass', num_classes=class_num).to(device)  # TODO: increase num classes if necessary 

        if self.parameter_optimization == 'CE':
            crit = CELoss(self.graph_model, self.aux, self.layer_wise)
        elif self.parameter_optimization == 'MI':
            crit = MILoss(self.aux, self.graph_model, self.MI_Y_LAMBDA, layer_wise=self.layer_wise)
        else:
            raise NotImplementedError(f"parameter optimization: {self.parameter_optimization}")

        if self.gamma_optimization  == 'CE':
            criterion2 = torch.nn.CrossEntropyLoss()
            crit2 = lambda out, int, targ: criterion2(out, targ)
        elif self.gamma_optimization == 'MI':
            crit2 = MILoss(self.aux,  self.graph_model, self.MI_Y_LAMBDA, layer_wise=False)
        else:
            raise NotImplementedError(
                f"gamma optimization: {self.gamma_optimization}")

        batch_seen = 0
        assert len(traindata) == len(valdata)
        for e in range(epoch_num):
            losses = []
            tq = tqdm.tqdm_notebook(zip(traindata, valdata))
            if self.layer_wise:
              crit.select_layer()
              optim = torch.optim.Adam(parameters, lr=lr)
            
            for (x, y), (x2,y2) in tq:
                optim.zero_grad()
                x = x.to(device)
                y = y.to(device)
                if intermediate_getter is None:
                    try:
                        out, intermediate = self.graph_model(x, intermediate=True)
                    except:
                        out = self.graph_model(x)
                        intermediate = self.graph_model.intermediate
                else:
                    out, intermediate = intermediate_getter(x)
                
               
                loss = crit(out, intermediate, y)
                loss.backward()
                optim.step()
                losses.append(loss.cpu().detach().numpy())
                tq.set_description(f'epoch: {e}. Loss: {str(np.mean(losses))}. Avg gamma: {str(torch.mean(abs(gammas[0])).item())}')
                
                x2 = x2.to(device)
                y2 = y2.to(device)
                optim2.zero_grad()
                if intermediate_getter is None:
                    out = self.graph_model(x)
                    intermediate = self.graph_model.intermediate
                else:
                    out, intermediate = intermediate_getter(x2)
                if not isinstance(out, torch.Tensor):
                    # when features are also returned in forward
                    out = out[0]

                loss2 = crit2(out, intermediate, y2)
                loss2.backward()
                optim2.step()
                
                batch_seen += 1
                if batch_seen % sample_mod == 0:
                    self.graph_model.eval()
                    for x, y in tqdm.tqdm_notebook(testdata):
                        x = x.to(device)
                        y = y.to(device)
                        out = self.graph_model(x)
                        if not isinstance(out, torch.Tensor):
                            # when features are also returned in forward
                            out = out[0]
                        pred = out.argmax(-1)
                        acc(pred, y)
                    accuracy = acc.compute().item()
                    print(
                        f'Epoch: {e}. Batch seen: {batch_seen}. Accuracy: {accuracy}')
                    history.append(accuracy)
                    acc.reset()
                    self.graph_model.train()
            

        return history
