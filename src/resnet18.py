import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18, self).__init__()
        self.selected_output = {}
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.fhooks = []
        for layer in self.model._modules.keys():
            self.fhooks.append(getattr(self.model, layer).register_forward_hook(self.forward_hook(layer)))

    def forward_hook(self, layer_name):
        def hook(module, inp, out):
            self.selected_output[layer_name] = out
        return hook

    def forward(self, x):
        return self.model(x), self.selected_output

