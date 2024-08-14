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
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret

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

    def forward(self, in_feat, adj):
        ori = in_feat
        print(in_feat.shape, adj)
        print(in_feat[:100])
        h = self.conv1(in_feat, adj) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self.conv2(h, adj) + self.residual * ori
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
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0], num_neg_samples=args.neg * pos_train_edge.shape[1])
    print(negedge.shape)
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor(h[edge[0]], h[edge[1]])
        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm] # try for now
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

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)
    pos_train_pred = []
    for perm in PermIterator(
            pos_train_edge.device, pos_train_edge.shape[0], batch_size, False
    ):
        edge = pos_train_edge[perm].t()
        pos_train_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_train_pred = torch.cat(pos_train_pred, dim=0)

    pos_valid_pred = []
    for perm in PermIterator(
            pos_valid_edge.device, pos_valid_edge.shape[0], batch_size, False
    ):
        edge = pos_valid_edge[perm].t()
        pos_valid_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_valid_pred = torch.cat(pos_valid_pred, dim=0)
    neg_valid_pred = []
    for perm in PermIterator(
            neg_valid_edge.device, neg_valid_edge.shape[0], batch_size, False
    ):
        edge = neg_valid_edge[perm].t()
        neg_valid_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    neg_valid_pred = torch.cat(neg_valid_pred, dim=0)

    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)
    pos_test_pred = []
    for perm in PermIterator(
            pos_test_edge.device, pos_test_edge.shape[0], batch_size, False
    ):
        edge = pos_test_edge[perm].t()
        pos_test_pred.append(
            predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        )
    pos_test_pred = torch.cat(pos_test_pred, dim=0)
    neg_test_pred = []
    for perm in PermIterator(
            neg_test_edge.device, neg_test_edge.shape[0], batch_size, False
    ):
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
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="ogbl-collab")
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--layers', type=int, default=1, help="layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--res_rate', type=float, default=0.5, help="residual connection rate")
    parser.add_argument('--dp', type=float, default=0.3, help="dropout ratio")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")
    parser.add_argument('--neg', type=int, default=1, help="number of negative samples")
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
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_edge = dataset.get_edge_split()
    data = data.to(device)
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        full_edge_index = add_self_loops(full_edge_index, num_nodes=data.num_nodes)[0]
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data.to(device)

    if args.dataset == "ogbl-ddi" or args.dataset == "ogbl-ppa":
        embedding = torch.nn.Embedding(data.num_nodes, args.hiddim).to(device)
        torch.nn.init.orthogonal_(embedding.weight)
        data.x = embedding.weight
    else:
        embedding = None

    predictor = Hadamard_MLPPredictor(args.hiddim, args.dp).to(device)
    
    ret = []

    set_seed(args.seed)
    bestscore = None
    bestepoch = 0
    
    # build model
    model = GCN(data.num_features, args.hiddim, args.layers, args.res_rate).to(device) 
        
    if args.dataset == "ogbl-ddi" or args.dataset == "ogbl-ppa":
        parameter = itertools.chain(model.parameters(), predictor.parameters(), embedding.parameters())
    else:
        parameter = itertools.chain(model.parameters(), predictor.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    for epoch in range(1, 1 + args.epochs):
        t1 = time.time()
        loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, embedding, args)
        scheduler.step()
        print(f"trn time {time.time()-t1:.2f} s", flush=True)
        t1 = time.time()
        results = test(model, predictor, data, split_edge, evaluator,
                        args.batch_size, args.use_valedges_as_input)
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