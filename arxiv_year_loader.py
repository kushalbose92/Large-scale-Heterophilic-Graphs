# from typing import Optional, Callable

import os.path as osp

import ogb
import torch
import numpy as np
import os

from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.nodeproppred import NodePropPredDataset


DATAPATH = '/splits/'

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def load_arxiv_year_dataset(root):
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv',root=root)
    graph = ogb_dataset.graph
    graph['edge_index'] = torch.as_tensor(graph['edge_index'])
    graph['node_feat'] = torch.as_tensor(graph['node_feat'])

    label = even_quantile_labels(graph['node_year'].flatten(), 5, verbose=False)
    label = torch.as_tensor(label).reshape(-1, 1)
    # split_idx_lst = load_fixed_splits("arxiv-year",os.path.join(root,"splits"))

    # train_mask = torch.stack([split["train"] for split in split_idx_lst],dim=1)
    # val_mask = torch.stack([split["valid"] for split in split_idx_lst],dim=1)
    # test_mask = torch.stack([split["test"] for split in split_idx_lst],dim=1)
    data = Data(x=graph["node_feat"],y=torch.squeeze(label.long()),edge_index=graph["edge_index"])
    return data


def load_fixed_splits(dataset,split_dir):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    splits_lst = np.load(os.path.join(split_dir,"{}-splits.npy".format(name)), allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


# root = os.getcwd() + '/splits/arxiv-year'
# data = load_arxiv_year_dataset(root)

# from torch_geometric.utils import homophily

# h_n = homophily(data.edge_index, data.y, method='edge')
# print(h_n)

# print(data.x.shape)
# print(data.edge_index.shape)
# print(data.train_mask.shape, "\t", data.val_mask.shape, "\t", data.test_mask.shape)
# print(data.y.shape)