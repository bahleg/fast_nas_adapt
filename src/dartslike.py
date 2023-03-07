import torch
import numpy as np
import torch
from torchmetrics import Accuracy
import tqdm
from torch.nn.functional import one_hot


class MILoss(torch.nn.Module):
    def __init__(self, aux, MI_Y_lambda = 0.0, num_classes=2, layer_wise: bool = False) -> None:
        super().__init__()
        self.aux = aux 
        self.MI_Y_lambda = MI_Y_lambda
        self.num_classes = num_classes
        self.layer_wise = layer_wise  # switch to layer wise optimization
        
    def forward(self, out, intermediate, target):
        ### intermediate 
        layers = list(self.aux.layer_names)
        loss = 0.0
        if self.layer_wise:
            selected_layer_ids = [np.random.randint(len(layers))]
        else:
            selected_layer_ids = range(len(layers))
        for i in range(len(layers)-1):
            if i not in selected_layer_ids:
                continue
            current_layer_name = layers[i]
            next_layer_name = layers[i+1]
            to_transform =  intermediate[current_layer_name].view(intermediate[current_layer_name].shape[0], -1)
            if self.layer_wise:
                to_transform = to_transform.detach()
            #print (current_layer_name, next_layer_name,to_transform.shape)
            #try:
            #    print (self.aux.means_int.weight.shape)
            #except:
            #    print ('low')
            mean = self.aux.means_int[current_layer_name](to_transform)
            log_sigma = self.aux.lsigmas_int[current_layer_name]
            loss += (log_sigma * np.prod(mean.shape) + \
                (((mean - intermediate[next_layer_name].view(intermediate[next_layer_name].shape[0], -1))**2)
                / (2 * torch.exp(log_sigma) ** 2))).sum() \
                    * (1.0-self.MI_Y_lambda)
        
        target = one_hot(target, self.num_classes)
         
        
        for i in range(len(layers)):
            if i not in selected_layer_ids:
                continue
            current_layer_name = layers[i]
            to_transform =  intermediate[current_layer_name].view(intermediate[current_layer_name].shape[0], -1)
            if self.layer_wise and i != len(layers) - 1:
                to_transform = to_transform.detach()
            if i == len(layers)-1:
                mean = to_transform 
                log_sigma = torch.zeros(1).to(to_transform.device)
            else:
                mean = self.aux.means_y[current_layer_name](to_transform)
                log_sigma = self.aux.lsigmas_y[current_layer_name]
            log_sigma = log_sigma.detach().view(1)  # detach sigma
            
            loss += (log_sigma * mean.numel()).sum()
            loss += (((mean - target) ** 2).sum() / (2 * torch.exp(log_sigma) ** 2)).sum()
            loss *= self.MI_Y_lambda
           
        return loss 


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
        

    def train_loop(self, traindata,  valdata, testdata, sample_mod, epoch_num, lr, lr2, device, wd):
        gammas =  [self.graph_model.gammas]
        parameters =  [p for n,p in self.graph_model.named_parameters() if n !='gammas']
        if self.gamma_optimization == 'MI':
            gammas.extend(list(self.aux.parameters()))
        if self.parameter_optimization == 'MI':
            parameters.extend(list(self.aux.parameters()))
        
        if not self.unrolled:
            optim = torch.optim.Adam(parameters, lr=lr)
            optim2 = torch.optim.Adam(gammas, lr=lr2, weight_decay=wd)
        else:
            raise NotImplementedError("unrolled")

        history = []
        acc = Accuracy(task='multiclass', num_classes=2, top_k=1).to(device)  # TODO: increase num classes if necessary
        if self.parameter_optimization == 'CE':
            criterion = torch.nn.CrossEntropyLoss()
            crit = lambda out, int, targ: criterion(out, targ)
        elif self.parameter_optimization == 'MI':
            crit = MILoss(self.aux, self.MI_Y_LAMBDA, layer_wise=self.layer_wise)
        else:
            raise NotImplementedError(
                f"parameter optimization: {self.parameter_optimization}")

        if self.gamma_optimization  == 'CE':
            criterion2 = torch.nn.CrossEntropyLoss()
            crit2 = lambda out, int, targ: criterion2(out, targ)

        elif self.gamma_optimization == 'MI':
            crit2 = MILoss(self.aux, self.MI_Y_LAMBDA, layer_wise=self.layer_wise)
        else:
            raise NotImplementedError(
                f"gamma optimization: {self.gamma_optimization}")

        batch_seen = 0
        assert len(traindata) == len(valdata)
        for e in range(epoch_num):
            losses = []
            tq = tqdm.tqdm_notebook(zip(traindata, valdata))
            for (x, y), (x2,y2) in tq:
                optim.zero_grad()
                x = x.to(device)
                y = y.to(device)
                out, intermediate = self.graph_model(x, intermediate=True)
                loss = crit(out, intermediate, y)
                loss.backward()
                optim.step()
                losses.append(loss.cpu().detach().numpy())
                tq.set_description(f'epoch: {e}. Loss: {str(np.mean(losses))}. Avg gamma: {str(torch.mean(abs(gammas[0])).item())}')

                x2 = x2.to(device)
                y2 = y2.to(device)
                optim2.zero_grad()
                out, intermediate = self.graph_model(x2, intermediate=True)
               
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
