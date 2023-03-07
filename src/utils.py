import tqdm

import numpy as np
import torch 
from torchmetrics import Accuracy

def test_loop(model, testdata, device):
    device = next(model.parameters()).device
    acc = Accuracy(task='multiclass', num_classes=2, top_k=1).to(device)  # TODO: adjust num_classes
    model.eval()
    for x,y in tqdm.tqdm_notebook(testdata):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        if not isinstance(out, torch.Tensor):
            out = out[0] #  when features are also returned in forward
        pred = out.argmax(-1)
        acc(pred, y)
    accuracy = acc.compute().item()
    
    model.train()
    return accuracy

    
    
    

def train_loop(model, traindata,  testdata, sample_mod, epoch_num, lr, device):
    history = []
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    batch_seen = 0
    for e in range(epoch_num):
        losses = []
        tq = tqdm.tqdm_notebook(traindata)
        for x,y in tq:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if not isinstance(out, torch.Tensor):
                out = out[0] #  when features are also returned in forward
            loss = crit(out, y)
            loss.backward()
            optim.step()
            losses.append(loss.cpu().detach().numpy())
            tq.set_description(f'epoch: {e}. Loss: {str(np.mean(losses))}')
            batch_seen += 1
            if batch_seen % sample_mod == 0:
                accuracy = test_loop(model, testdata, device)
                print (f'Epoch: {e}. Batch seen: {batch_seen}. Accuracy: {accuracy}')
                history.append(accuracy)
                
    return history
