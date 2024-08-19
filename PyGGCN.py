import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
import time
from torch_geometric.nn import GCNConv
from torch.optim import lr_scheduler
import itertools
from torch_geometric.utils import add_self_loops, to_undirected
from torch_sparse import SparseTensor

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

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
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.prop_step = prop_step
        self.residual = residual

    def forward(self, in_feat, ei):
        ori = in_feat
        h = self.conv1(in_feat, ei) + self.residual * ori
        for _ in range(1, self.prop_step):
            h = self.conv2(h, ei) + self.residual * ori
        return h


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          embedding, 
          args):
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    loader = torch.utils.data.DataLoader(range(pos_train_edge.shape[1]), batch_size=batch_size, shuffle=True)
    neg_edge = negative_sampling(data.edge_index, data.num_nodes, num_neg_samples=args.neg * pos_train_edge.shape[1])
    for perm in loader:
        optimizer.zero_grad()
        h = model(data.x, data.adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor(h[edge[0]], h[edge[1]])
        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = neg_edge[:, perm]
        neg_outs = predictor(h[edge[0]], h[edge[1]])
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        if args.dataset == "ogbl-ddi":
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            nn.utils.clip_grad_norm_(embedding.parameters(), 1.0)
        elif args.dataset != "ogbl-ppa":
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj.device())

    h = model(data.x, data.adj)
    pos_train_pred = []
    loader = torch.utils.data.DataLoader(range(pos_train_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_train_edge[perm].t()
        pos_train_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_train_pred = torch.cat(pos_train_pred, dim=0)

    pos_valid_pred = []
    loader = torch.utils.data.DataLoader(range(pos_valid_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_valid_edge[perm].t()
        pos_valid_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_valid_pred = torch.cat(pos_valid_pred, dim=0)
    neg_valid_pred = []
    loader = torch.utils.data.DataLoader(range(neg_valid_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = neg_valid_edge[perm].t()
        neg_valid_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    neg_valid_pred = torch.cat(neg_valid_pred, dim=0)

    if use_valedges_as_input:
        h = model(data.x, data.full_adj)
    pos_test_pred = []
    loader = torch.utils.data.DataLoader(range(pos_test_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_test_edge[perm].t()
        pos_test_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_test_pred = torch.cat(pos_test_pred, dim=0)
    neg_test_pred = []
    loader = torch.utils.data.DataLoader(range(neg_test_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = neg_test_edge[perm].t()
        neg_test_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    neg_test_pred = torch.cat(neg_test_pred, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="ogbl-collab")
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--layers', type=int, default=1, help="layers")
    parser.add_argument('--hidden', type=int, default=32, help="hidden dimension")
    parser.add_argument('--residual', type=float, default=0, help="residual connection rate")
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout ratio")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")
    parser.add_argument('--num_neg', type=int, default=1, help="number of negative samples")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    #print(args, flush=True)

    evaluator = Evaluator(name=args.dataset)

    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    data.edge_index = to_undirected(data.edge_index)
    data.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    split_edge = dataset.get_edge_split()
    data.neg = negative_sampling(data.edge_index.to(device), data.num_nodes, num_neg_samples=args.num_neg * split_edge['train']['edge'].shape[1])
    data = data.to(device)
    if args.data == 'ogbl-collab':
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        full_edge_index = add_self_loops(full_edge_index, num_nodes=data.num_nodes)[0]
        full_edge_index = to_undirected(full_edge_index)
        data.full_adj = SparseTensor(row=full_edge_index[0], col=full_edge_index[1])
    else:
        data.full_adj = data.adj

    data = data.to(device)

    if args.dataset == "ogbl-ddi" or args.dataset == "ogbl-ppa":
        embedding = torch.nn.Embedding(data.num_nodes, args.hidden).to(device)
        torch.nn.init.orthogonal_(embedding.weight)
        data.x = embedding.weight
    else:
        embedding = None

    predictor = Hadamard_MLPPredictor(args.hidden, args.dropout).to(device)
    
    ret = []

    set_seed(args.seed)
    bestscore = None
    bestepoch = 0
    
    # build model
    model = GCN(data.num_features, args.hidden, args.layers, args.residual).to(device) 
        
    if args.dataset == "ogbl-ddi" or args.dataset == "ogbl-ppa":
        parameter = itertools.chain(model.parameters(), predictor.parameters(), embedding.parameters())
    else:
        parameter = itertools.chain(model.parameters(), predictor.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr)
    
    for epoch in range(1, 1 + args.epochs):
        t1 = time.time()
        loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, embedding, args)
        if epoch % 100 == 0:
            adjustlr(optimizer, epoch / args.epochs, args.lr)
        print(f"trn time {time.time()-t1:.2f} s", flush=True)
        t1 = time.time()
        use_valedges_as_input = (args.dataset == 'ogbl-collab')
        results = test(model, predictor, data, split_edge, evaluator,
                        args.batch_size, use_valedges_as_input)
        print(f"test time {time.time()-t1:.2f} s")
        if bestscore is None:
            bestscore = {key: list(results[key]) for key in results}

        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            if args.dataset == 'ogbl-ddi':
                if valid_hits > bestscore[key][1] and epoch >= 300:
                    bestscore[key] = list(result)
                    bestepoch = epoch
            else:
                if valid_hits > bestscore[key][1]:
                    bestscore[key] = list(result)
                    bestepoch = epoch
                if bestepoch + 100 < epoch:
                    break

            print(key)
            loss = 0
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_hits:.2f}%, '
                    f'Valid: {100 * valid_hits:.2f}%, '
                    f'Test: {100 * test_hits:.2f}%')
        print('---', flush=True)
    print(f"best {bestscore}")
    if args.dataset == "ogbl-collab":
        ret.append(bestscore["Hits@50"][-2:])
    elif args.dataset == "ogbl-ppa":
        ret.append(bestscore["Hits@100"][-2:])
    elif args.dataset == "ogbl-ddi":
        ret.append(bestscore["Hits@20"][-2:])
    elif args.dataset == "ogbl-citation2":
        ret.append(bestscore[-2:])
    else:
        raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")

if __name__ == "__main__":
    main()