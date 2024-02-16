import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from CSSM import CSSMB


class FDV_Net(nn.Module):
    def __init__(self, n_blocks=8):
        super().__init__()
        self.feats = nn.Sequential(*[CSSMB() for _ in range(n_blocks)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=True) 
        output = self.feats(x1)
        output = F.interpolate(output, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        return self.sigmoid(output) + x
    

#data = torch.randn(1, 3, 64, 64).cuda()

#model = FDV_Net().cuda()

#output = model(data)



#print(output.shape)