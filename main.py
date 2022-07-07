from collections import namedtuple
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import time
import os 
from datetime import datetime
import pickle
import gc
import copy

import torch as th
import dgl
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from initializer import initialize_model,initialize_optimizer
from dataset import TwitterDataset, batcher, create_dataset
from loss import loss_fn

def train_loop(model,data_loader,optimizer,device,h_size,beta,gamma):
    train_preds = []
    train_true_l = []
    train_logits = []

    model.train()
    for step, batch in tqdm(enumerate(data_loader),total=len(data_loader)):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'train')

        true_labels = batch.label[batch.train_mask==1] 

        loss = loss_fn(logits, true_labels, 2, true_labels.unique(return_counts=True)[1].tolist(), device, beta, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = th.argmax(logits, 1)

        train_logits.append(logits)
        train_preds.extend(pred.to('cpu'))
        train_true_l.extend(true_labels.to('cpu'))

    train_metrics = model.compute_metrics(train_true_l,train_preds)
    train_logits = th.cat(train_logits).to(device)
    train_true_l = th.tensor(train_true_l).to(device)
    train_loss = loss_fn(train_logits, train_true_l, 2, train_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)

    print("Train Loss {:.4f} | ".format(train_loss.item()),end='')


def val_loop(model,data_loader,device,h_size,beta,gamma):
    val_preds = []
    val_true_l = []
    val_logits = []
    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'val')
        val_logits.append(logits)
        true_labels = batch.label[batch.val_mask==1] 
        pred = th.argmax(logits, 1)
        val_preds.extend(pred.to('cpu'))
        val_true_l.extend(true_labels.to('cpu'))

    val_metrics = model.compute_metrics(val_true_l,val_preds)
    val_logits = th.cat(val_logits).to(device)
    val_true_l = th.tensor(val_true_l).to(device)
    val_loss = loss_fn(val_logits, val_true_l, 2, val_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)
    val_metrics["loss"] = val_loss.item()

    print("Val Loss {:.4f} |".format(val_loss.item()))
    
    return val_metrics

def test_loop(model,data_loader,device,h_size,beta,gamma,save=False):
    test_preds = []
    test_true_l = []
    test_logits = []
    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'test',save=save)
        #logits = logits[batch.test_mask==1]
        test_logits.append(logits)
        true_labels = batch.label[batch.test_mask==1] 
        pred = th.argmax(logits, 1)
        test_preds.extend(pred.to('cpu'))
        test_true_l.extend(true_labels.to('cpu'))

    if save:
        model.save_embs()
        model.save_data = []
        gc.collect()

    test_metrics = model.compute_metrics(test_true_l,test_preds)
    test_logits = th.cat(test_logits).to(device)
    test_true_l = th.tensor(test_true_l).to(device)
    test_loss = loss_fn(test_logits, test_true_l, 2, test_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)
    test_metrics["loss"] = test_loss.item()
    return test_metrics

def main(args, params=None):
    start = time.time()

    if not args:
        data_dir = params["data-dir"]
        x_size = params["x-size"]
        h_size = params["h-size"]
        dropout = params["dropout"]
        lr = params["lr"]
        weight_decay = params["weight-decay"]
        epochs = params["epochs"]
        beta = params["beta"]
        gamma = params["gamma"]
        batch_size = params["batch-size"]
        patience = params["patience"]
        min_epochs = params["min-epochs"]
        device = params["device"]
        optim_type = params["optimizer"]
        save = params["save"]
        save_dir = params["save-dir"]
        print(params)
    else:
        data_dir = args.data_dir
        x_size = args.x_size
        h_size = args.h_size
        dropout = args.dropout
        lr = args.lr
        weight_decay = args.weight_decay
        epochs = args.epochs
        beta = args.beta
        gamma = args.gamma
        batch_size = args.batch_size
        patience = args.patience
        min_epochs = args.min_epochs
        device = args.device
        optim_type = args.optimizer
        save = args.save
        save_dir = args.save_dir
        print(args)


    train_dataset, val_dataset, test_dataset = create_dataset(data_dir) 
    train_trees = train_dataset.trees
    val_trees = val_dataset.trees
    test_trees = test_dataset.trees

    num_classes = train_dataset.num_classes

    if device=='auto':
        device = 'cuda' if th.cuda.is_available() else 'cpu'

    print("Device:",device)

    if not args:
        model = initialize_model(num_classes, device, None, params)
    else:
        model = initialize_model(num_classes, device, args)
    model.to(device)
    print(model)

    optimizer = initialize_optimizer(optim_type)(model.parameters(),lr=lr,weight_decay=weight_decay)

    train_loader = DataLoader(dataset=train_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    val_loader = DataLoader(dataset=val_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    test_loader = DataLoader(dataset=test_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)

    counter=0
    best_val_metrics = model.init_metric_dict()
    test_metrics = None
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        print("Epoch: ",epoch)
        train_loop(model,train_loader,optimizer,device,h_size,beta,gamma)

        val_metrics = val_loop(model,val_loader,device,h_size,beta,gamma)
        if model.has_improved(best_val_metrics, val_metrics):
            test_metrics = test_loop(model,test_loader,device,h_size,beta,gamma)
            best_val_metrics = val_metrics
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0 
        else:
            counter += 1
            if counter == patience and epoch > min_epochs:
                print("Early stopping")
                break

    print("(Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(test_metrics["loss"], test_metrics["f1"], test_metrics["recall"]))
    print(test_metrics["conf_mat"])

    if not os.path.exists('results'):
        os.makedirs('results')

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    end = time.time()
    print("Time Elapsed: ",end-start)

    save_dict = {}
    save_dict['test_metrics'] = test_metrics
    save_dict['args'] = args
    save_dict['params'] = params
    save_dict['time'] = current_time

    dir_ = 'results/'+save_dir+'/'+current_time

    if save:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = dir_ + "/" + current_time + ".pkl"

        with open(fname,'wb') as f:
            pickle.dump(save_dict,f)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--x-size', type=int, default=768)
    parser.add_argument('--h-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--min-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5.75e-4)
    parser.add_argument('--weight-decay', type=float, default=5.75e-6)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('-beta', '--beta', default=0.999999, type=float)
    parser.add_argument('-gamma', '--gamma', default=2.5, type=float)
    parser.add_argument('--data-dir', type=str, default='./data',help='directory for data')
    parser.add_argument('--optimizer', type=str, default='Adam',choices=['Adam','RiemannianAdam'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-dir', type=str, default='res',help='save directory')
    parser.add_argument('--c', type=float, default=1.0)

    args = parser.parse_args()
    
    main(args)