import pretrainedmodels
import torch
import torch.nn as nn


class model_SeResNeXt50(nn.Module):

    def __init__(self, num_classes, inference, freeze_encoder, **kwargs):
        super(model_SeResNeXt50, self).__init__()

        self.encoder = pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained='imagenet' if not inference else None)
        self.global_pool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential()
        self.classifier.add_module('proj', nn.Linear(2048, num_classes))
        if inference:
            self.classifier.add_module('softmax', nn.Softmax())

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)

        if freeze_encoder and not inference:
            print('Freezing encoder layers')
            self.freeze_module([self.encoder])

    def forward(self, batch):
        rgb = batch['rgb']

        self.img_mean = self.img_mean.to(rgb.device)
        self.img_std = self.img_std.to(rgb.device)

        # normalizing image
        rgb = (rgb - self.img_mean) / self.img_std

        features = self.encoder.features(rgb)
        pooled = self.global_pool(features).view(features.size(0), -1)
        probs = self.classifier(pooled)
        return probs

    def freeze_module(self, module_list: list):
        for mod in module_list:
            for param in mod.parameters():
                param.requires_grad = False
