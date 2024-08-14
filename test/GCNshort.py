import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, residual=0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.prop_step = prop_step
        self.residual = residual

    def forward(self, g, in_feat):
        ori = in_feat
        h = self.conv1(g, in_feat) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self.conv2(g, h) + self.residual * ori
        return h

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-ddi', choices=['ogbl-ddi', 'ogbl-ppa'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=False)
    parser.add_argument("--metric", default='hits@20', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--residual", default=0, type=float)
    args = parser.parse_args()
    return args

args = parse()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dgl.seed(args.seed)

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        h = model(g, g.ndata['feat'])

        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge
        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        
        optimizer.zero_grad()
        loss.backward()
        if args.dataset == 'ogbl-ddi':
            torch.nn.utils.clip_grad_norm_(g.ndata['feat'], 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, evaluator, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
        dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_test_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_test_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        pos_score = torch.reshape(pos_score, (pos_score.shape[0], 1))
        neg_score = torch.reshape(neg_score, (neg_score.shape[0], 1))
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

def eval(model, g, pos_valid_edge, neg_valid_edge, evaluator, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
        dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_valid_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_valid_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        pos_score = torch.reshape(pos_score, (pos_score.shape[0], 1))
        neg_score = torch.reshape(neg_score, (neg_score.shape[0], 1))
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]
graph = dgl.add_self_loop(graph)
graph = dgl.to_bidirected(graph, copy_ndata=True).to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

pred = Hadamard_MLPPredictor(args.hidden, args.dropout).to(device)

embedding = torch.nn.Embedding(graph.num_nodes(), args.hidden).to(device)
torch.nn.init.orthogonal_(embedding.weight)
graph.ndata['feat'] = embedding.weight

model = GCN(graph.ndata['feat'].shape[1], args.hidden, args.prop_step, args.residual).to(device)

parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0

losses = []
valid_list = []
test_list = []

for epoch in range(args.epochs):
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred)
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results = eval(model, graph, valid_pos_edge, valid_neg_edge, evaluator, pred)
    valid_list.append(valid_results[args.metric])
    test_results = test(model, graph, test_pos_edge, test_neg_edge, evaluator, pred)
    test_list.append(test_results[args.metric])
    if valid_results[args.metric] > best_val:
        if args.dataset == 'ogbl-ddi' and epoch > 300:
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
        elif args.dataset != 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
    if args.dataset == 'ogbl-ppa':
        if best_epoch + 100 < epoch:
            break

    print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation hit: {valid_results[args.metric]:.4f}, Test hit: {test_results[args.metric]:.4f}")

print(f"Test hit: {final_test_result[args.metric]:.4f}")