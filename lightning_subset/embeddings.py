import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.hub import load_state_dict_from_url
import torchvision


class BaseImageEmbedding(nn.Module):
    def __init__(self, model, img_res=224):
        super().__init__()

        self.model = model
        self.img_res = img_res
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        # expects image in range [-1, 1]
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std

        if (x.shape[2] != self.img_res) or (x.shape[3] != self.img_res):
            x = F.interpolate(x,
                              size=(self.img_res, self.img_res),
                              mode='bilinear',
                              align_corners=True)

        x = self.model(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
        
class Inceptionv3Embedding(BaseImageEmbedding):
    def __init__(self):
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = Identity()
        super().__init__(model, img_res=299)


