import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import pickle


from models.base_model import BaseModel
from models.bu_lstm_cells import HCN


class UniTreeLSTM(BaseModel):
    def __init__(self, x_size, h_size, num_classes, dropout, device, c=1.0):
        super().__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.root_linear = nn.Linear(x_size,h_size//2)
        self.linear1 = nn.Linear(h_size, h_size//2)
        self.linear2 = nn.Linear(h_size, num_classes)

        cell = HCN
        cell = cell(x_size, h_size, device, c)

        self.cell = cell
        self.device = device

        self.save_data = []

    def forward(self, batch, h, c, mode='train',save=False):
        g = batch.graph.to(self.device)

        embeds = batch.feats.to(self.device)

        g.ndata['iou1'] = self.cell.pmath_geo.mobius_matvec(self.cell.W_iou,self.cell.pmath_geo.expmap0(self.dropout(embeds)))
        g.ndata['h1'] = h
        g.ndata['c1'] = c
        
        dgl.prop_nodes_topo(g, message_func=self.cell.message_func, reduce_func=self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)

        h = self.dropout(self.cell.pmath_geo.logmap0(g.ndata.pop('h1')))
        h = self.linear1(h)
        
        if mode=='train':
            mask = batch.train_mask
        elif mode=='val':
            mask = batch.val_mask
        elif mode == 'test':
            mask = batch.test_mask

        if save:
            g.ndata['embs'] = h
        
        h = h[mask==1]

        root_feat = batch.feats[mask==1]
        root_feat = self.root_linear(root_feat)
        h = torch.cat((root_feat,h),dim=-1)
        logits = self.linear2(h)

        if save:
            self.save_data.append((g.cpu(),logits,g.ndata['y']))

        return logits
        
    def save_embs(self):
        with open("embeddings.pkl",'wb') as f:
            pickle.dump(self.save_data,f)