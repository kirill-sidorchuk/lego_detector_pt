import torch
import torchvision.models as models
import torch.nn as nn


class model_ResNet50(nn.Module):

    def __init__(self, num_classes, inference):
        super().__init__()

        self.encoder = models.resnet50(pretrained=True)
        self.classifier = nn.Sequential()
        self.classifier.add_module('proj', nn.Linear(2048, num_classes))
        if inference:
            self.classifier.add_module('softmax', nn.Softmax())

        # self.freeze_module([self.encoder])

    def forward(self, batch):
        rgb = batch['rgb']

        x = self.encoder.conv1(rgb)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)

        probs = self.classifier(x)
        return probs

    def freeze_module(self, module_list: list):
        for mod in module_list:
            for param in mod.parameters():
                param.requires_grad = False
