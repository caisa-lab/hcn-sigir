from models.uni_model import UniTreeLSTM
import torch
from geoopt.optim.radam import RiemannianAdam

def initialize_model(num_classes, device, args, params=None):
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
        c = params["c"]
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
        c = args.c
  
    model = UniTreeLSTM(x_size, h_size, num_classes, dropout, device, c)

    return model

def initialize_optimizer(type='Adam'):
    if type == 'Adam':
        return torch.optim.Adam
    elif type == 'RiemannianAdam':
        return RiemannianAdam
    return None
