import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout):
        super().__init__()
        self.W1 = nn.Linear(h_feats, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.dropout = dropout

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.W1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W2(x)
        return x.squeeze()

class att_Hadamard(nn.Module):
    def __init__(self, h_feats, dropout):
        super().__init__()
        self.pred = Hadamard_MLPPredictor(h_feats, dropout)
        self.att = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x_i, x_j, y_i, y_j):
        x = self.pred(x_i, x_j)
        y = self.pred(y_i, y_j)
        return self.att * x + (1 - self.att) * y
        

def drop_edge(g, dpe = 0.2):
    g = g.clone()
    eids = torch.randperm(g.number_of_edges())[:int(g.number_of_edges() * dpe)].to(g.device)
    g.remove_edges(eids)
    g = dgl.add_self_loop(g)
    return g
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0.2, drop_edge=False, relu=False, prop_step=2, residual=0, K=0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        if K > 0:
            self.conv_neg_1 = GraphConv(in_feats, h_feats)
            self.conv_neg_2 = GraphConv(h_feats, h_feats)
            self.K = K
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        if norm:
            self.ln = nn.LayerNorm(h_feats)
            self.dp = nn.Dropout(dp4norm)

    def forward(self, g, in_feat, neg_g=None):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        if neg_g is not None:
            h = self.conv1(g, in_feat) - 0.1 * self.conv_neg_1(neg_g, in_feat) / self.K + self.residual * ori
        else:
            h = self.conv1(g, in_feat) + self.residual * ori
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
            if self.norm:
                h = self.ln(h)
                h = self.dp(h)
            if neg_g is not None:
                h = self.conv2(g, h) - 0.1 * self.conv_neg_2(neg_g, h) / self.K + self.residual * ori
            else:
                h = self.conv2(g, h) + self.residual * ori
        return h

class GCN_with_feature(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step = 2, dropout = 0.2, residual=0):
        super(GCN_with_feature, self).__init__()
        self.conv1 = GraphConv(h_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv1_feat = GraphConv(in_feats, h_feats)
        self.conv2_feat = GraphConv(h_feats, h_feats)
        self.mlp = nn.Sequential(
            nn.Linear(2 * h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.prop_step = prop_step
        self.residual = residual

    def forward(self, g, in_feat, in_feat2):
        x = in_feat
        h = self.conv1(g, in_feat) + self.residual * x
        h2 = self.conv1_feat(g, in_feat2)
        h2 = F.relu(h2)
        for _ in range(1, self.prop_step):
            h = self.conv2(g, h) + self.residual * x
            h2 = self.conv2_feat(g, h2) + self.residual
            h2 = F.relu(h2)
        h = self.mlp(torch.cat([h, h2], dim=1))
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