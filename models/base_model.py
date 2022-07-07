import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, batch, h, c):
        raise NotImplementedError

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1, 'recall': -1, 'loss': np.inf}

    def has_improved(self, m1, m2):
        return m1["loss"] > m2['loss']

    def compute_metrics(self,true_l,preds):

        report = classification_report(true_l, preds, labels=[0, 1], output_dict=True)
        rec = report["1"]["recall"]
        m_f1 = report["macro avg"]["f1-score"]
        conf_mat = confusion_matrix(true_l,preds)

        metrics = {"f1":m_f1,"recall":rec,"conf_mat":conf_mat}
        return metrics