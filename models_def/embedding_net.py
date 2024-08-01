import torchvision
import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # based on the ResNet18 CNN
        self.resnet = torchvision.models.resnet18(weights=None)
        self.fc_in_features = self.resnet.fc.in_features
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(nn.Linear(self.fc_in_features, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                )

    def forward(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
