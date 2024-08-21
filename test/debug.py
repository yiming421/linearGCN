from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import add_self_loops, to_undirected
import torch.nn as nn

def test():
    ei = torch.tensor([[2, 3, 4], [1, 2, 3]]).cuda(0)
    sp = SparseTensor.from_edge_index(ei, sparse_sizes=(5, 5))
    model = GCNConv(2, 2).cuda(0)
    x = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]).float().cuda(0)
    print(x, sp)
    model(x, sp)

test()
