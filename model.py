import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv



class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res

    def forward(self, x_i, x_j):
        x = x_i * x_j
        ori = x
        for lin in self.lins[:-1]:
            x = lin(x)
            if self.res:
                x = x + ori
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()

class DotPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        x = (x_i * x_j).sum(dim=-1)
        return x.squeeze()
    
class LorentzPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        n = x_i.size(1)
        x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
        return x.squeeze()

def drop_edge(g, dpe = 0.2):
    g = g.clone()
    eids = torch.randperm(g.number_of_edges())[:int(g.number_of_edges() * dpe)].to(g.device)
    g.remove_edges(eids)
    g = dgl.add_self_loop(g)
    return g
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0.2, drop_edge=False, relu=False, prop_step=2, residual=0, conv='GCN'):
        super(GCN, self).__init__()
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.conv1 = GINConv(nn.Linear(in_feats, h_feats), 'mean')
            self.conv2 = GINConv(nn.Linear(h_feats, h_feats), 'mean')
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        if norm:
            self.ln = nn.LayerNorm(h_feats)
            self.dp = nn.Dropout(dp4norm)

    def forward(self, g, in_feat):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
            if self.norm:
                h = self.ln(h)
                h = self.dp(h)
            h = self.conv2(g, h).flatten(1) + self.residual * ori
        return h

class GCN_with_feature(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, residual = 0, relu = False, conv='GCN'):
        super(GCN_with_feature, self).__init__()
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.conv1 = GINConv(nn.Linear(in_feats, h_feats), 'mean')
            self.conv2 = GINConv(nn.Linear(h_feats, h_feats), 'mean')
        self.prop_step = prop_step
        self.residual = residual
        self.dp = dropout
        self.relu = relu

    def forward(self, g, in_feat, e_feat=None):
        h = self.conv1(g, in_feat, edge_weight=e_feat).flatten(1)
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
                h = F.dropout(h, p=self.dp, training=self.training)
            h = self.conv2(g, h, edge_weight=e_feat).flatten(1)
        return h

class GCN_with_MLP(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.2, prop_step = 2, relu = False):
        super(GCN_with_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv = GraphConv(h_feats, h_feats)
        self.prop_step = prop_step
        self.relu = relu

    def forward(self, g, in_feat):
        h = self.mlp(in_feat)
        for i in range(self.prop_step):
            if self.relu:
                h = F.relu(h)
            h = self.conv(g, h)
        return h
    
class GCN_no_para(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.2, prop_step = 2, relu = False):
        super(GCN_no_para, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv = GraphConv(h_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.relu = relu

    def forward(self, g, in_feat):
        h = self.mlp(in_feat)
        for i in range(self.prop_step):
            if self.relu:
                h = F.relu(h)
            h = self.conv(g, h)
        return h