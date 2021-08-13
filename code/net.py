import torch.nn as nn

import torchvision

class MyNet(nn.Module):
    def __init__(self, category_num):
        super(MyNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.end = nn.Linear(1000, category_num, bias=True)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.end(x)
        
        return x
    

