import dgl
import pickle
import numpy as np
import os
import torch
import networkx as nx
from collections import namedtuple


TreeBatch = namedtuple('TreeBatch', ['graph', 'feats', 'label', 'del_t', 'train_mask', 'val_mask', 'test_mask'])

def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return TreeBatch(graph=batch_trees,train_mask=batch_trees.ndata["train_mask"].to(device),val_mask=batch_trees.ndata["val_mask"].to(device),
            test_mask=batch_trees.ndata["test_mask"].to(device),feats=batch_trees.ndata['x'].to(device),label=batch_trees.ndata['y'].to(device),
            del_t=batch_trees.ndata['del_t'])
    return batcher_dev

class TwitterProcessor():
    def __init__(self, data_dir):

        self.attrs = ['x', 'y', 'del_t', 'train_mask', 'val_mask', 'test_mask']

        filepath = os.path.join(data_dir,"data_sample.pkl")
        with open(filepath,"rb") as f:
            dataset = pickle.load(f)
        self.trees = dataset
        self.labels = None
        self.num_classes = 2

    def __getitem__(self,item):
        return self.trees[item]
    
    def __len__(self):
        return len(self.trees)
    
    def get_labels(self):
        return self.labels

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self,trees,labels):
        
        self.trees = trees
        self.labels = labels
        self.num_classes = 2

    def __getitem__(self,item):
        return self.trees[item]
    
    def __len__(self):
        return len(self.trees)
    
    def get_labels(self):
        return self.labels


def create_dataset(data_dir):
    processor = TwitterProcessor(data_dir)
    train_trees = []
    train_labels = []
    val_trees = []
    val_labels = []
    test_trees = []
    test_labels = []

    if processor.labels:
        for t, l in zip(processor.trees, processor.labels):
            if(sum(t.ndata['train_mask'])==1):
                train_trees.append(t)
                train_labels.append(l)
            if(sum(t.ndata['val_mask'])==1):
                val_trees.append(t)
                val_labels.append(l)
            if(sum(t.ndata['test_mask'])==1):
                test_trees.append(t)
                test_labels.append(l)
    else:
        for t in processor.trees:
            if(sum(t.ndata['train_mask'])==1):
                train_trees.append(t)
            if(sum(t.ndata['val_mask'])==1):
                val_trees.append(t)
            if(sum(t.ndata['test_mask'])==1):
                test_trees.append(t)
    print(len(train_trees),len(val_trees),len(test_trees))
    return TwitterDataset(train_trees,train_labels), TwitterDataset(val_trees,val_labels), TwitterDataset(test_trees,test_labels)
