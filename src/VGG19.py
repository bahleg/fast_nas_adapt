import torch
import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class VGG19(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG19, self).__init__()
        self.selected_output = {}
        self.model = vgg19_bn()
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.fhooks = []
        for layer in self.model._modules.keys():
            self.fhooks.append(getattr(self.model, layer).register_forward_hook(self.forward_hook(layer)))

    def forward_hook(self, layer_name):
        def hook(module, inp, out):
            self.selected_output[layer_name] = out
        return hook

    def forward(self, x):
        return self.model(x), self.selected_output

