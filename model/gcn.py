import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_scatter import scatter


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, fea_nums, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('bij, jk -> bik', input, self.weight)

        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, node_nums, cluster_num=100, dropout=0.2):
        super(GCN, self).__init__()
        '''
        nfeat:feature dimension
        nclass：number of classes
        node_nums：number of graph nodes
        '''
        self.gc1 = GraphConvolution(nfeat, nfeat * 2, node_nums)
        self.gc2 = GraphConvolution(nfeat * 2, nfeat * 4, node_nums)
        self.gc3 = GraphConvolution(nfeat * 4, 16, node_nums)
        self.dropout = dropout
        self.cluster_num = cluster_num

        self.concat_and_classify = nn.Sequential(
            nn.Linear(16 * cluster_num, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, nclass),
        )

    def _reset_parameters(self):
        for layer in self.concat_and_classify.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                # torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val=1.0)
                torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x, adj, mask, in_node_num, groups):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)

        x = scatter(x, groups.long(), dim=1, reduce="max")

        b, node_num, fea_dim = x.shape
        in_node_num = in_node_num.unsqueeze(-1)
        in_node_num = in_node_num.repeat(1, 1, fea_dim)
        x = x * in_node_num

        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, fea_dim)
        x = x * mask
        x = x.view(b, -1)

        x = self.concat_and_classify(x)

        return x


class GCN_Binary(nn.Module):
    def __init__(self, nfeat, node_nums, fc_in_dim=16, cluster_num=100, dropout=0.2):
        super(GCN_Binary, self).__init__()
        '''
        nfeat: feature dimension
        nclass: number of classes
        node_nums: number of graph nodes
        '''

        self.gc1 = GraphConvolution(nfeat, nfeat * 2, node_nums)
        self.gc2 = GraphConvolution(nfeat * 2, nfeat * 4, node_nums)
        self.gc3 = GraphConvolution(nfeat * 4, fc_in_dim, node_nums)
        self.dropout = dropout
        self.fc_in_dim = fc_in_dim

        self.concat_and_classify = nn.Sequential(
            nn.Linear(self.fc_in_dim * cluster_num, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )

    def _reset_parameters(self):
        for layer in self.concat_and_classify.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val=1.0)
                torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x, adj, mask, in_node_num, groups):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)

        x = scatter(x, groups.long(), dim=1, reduce="max")

        b, node_num, fea_dim = x.shape
        in_node_num = in_node_num.unsqueeze(-1)
        in_node_num = in_node_num.repeat(1, 1, fea_dim)
        x = x * in_node_num

        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, fea_dim)
        x = x * mask
        x = x.view(b, -1)
        x = self.concat_and_classify(x)

        x = torch.sigmoid(x)

        return x

if __name__ == '__main__':
    pass
