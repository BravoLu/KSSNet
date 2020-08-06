import torchvision.models as models
from torch.nn import Parameter
#from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *

class GraphConvolution(nn.Module):
    """
    	Source: "https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py"
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LC(nn.Module):
    def __init__(self, hid_dim, num_classes=80):
        super(LC, self).__init__()
        self.conv1x1 = nn.Conv2d(num_classes, hid_dim, kernel_size=1, stride=1, bias=False)
        self.conv1x1.apply(weights_init)

    def forward(self, x, E):
        N, C, H, W = x.size()
        original_x = x
        # [N, C, H, W] -> [N, HW, C]
        x = x.transpose(1,2).transpose(2,3).view(N, H*W, -1)
        # E = [N, C] -> [C, N]
        E = E.transpose(1, 0)
        E = torch.tanh(E)
        x = torch.matmul(x, E)
        x = x.view(N, H, W, -1)
        x = x.transpose(3, 1).transpose(2, 3)
        x = self.conv1x1(x)
        x = x + original_x
        return x


class KSSNet(nn.Module):
    def __init__(self, ):
        super(KSSNet, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.res_block1 = self.backbone.layer1
        self.res_block2 = self.backbone.layer2
        self.res_block3 = self.backbone.layer3
        self.res_block4 = self.backbone.layer4

        self.gcn1 = GraphConvolution(256, 256)
        self.gcn2 = GraphConvolution(256, 512)
        self.gcn3 = GraphConvolution(512, 1024)
        self.gcn4 = GraphConvolution(1024, 2048)

        self.lc1 = LC(256)
        self.lc2 = LC(512)
        self.lc3 = LC(1024)
        self.lc4 = LC(2048)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.up_layer = nn.Conv2d(2048, 80, 1, 1)

    def forward(self, x, word_embedding):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        adj = torch.rand((80,80))
        x = self.res_block1(x)
        e = self.gcn1(word_embedding, adj)
        import pdb
        #pdb.set_trace()
        x = self.lc1(x, e)
        e = F.leaky_relu(e, 0.2)


        x = self.res_block2(x)
        e = self.gcn2(e, adj)
        x = self.lc2(x, e)
        e = F.leaky_relu(e, 0.2)

        x = self.res_block3(x)
        e = self.gcn3(e, adj)
        x = self.lc3(x, e)
        e = F.leaky_relu(e, 0.2)

        x = self.res_block4(x)
        e = self.gcn4(e, adj)
        x = self.lc4(x, e)
        e = torch.sigmoid(e)

        feat = self.gap(x)
        x = feat.view(feat.size(0), -1)
        # e = [feat, label_number]
        e = e.transpose(0,1)
        x = torch.matmul(x, e)
        y = self.up_layer(feat)
        y = y.view(y.size(0), -1)
        x = x + y
        return x

if __name__ == "__main__":
    kssnet = KSSNet()
    print(kssnet.backbone.modules())
    x = torch.rand((32, 3, 224, 224))
    word_embedding = torch.rand((80, 256))
    out = kssnet(x,word_embedding)
