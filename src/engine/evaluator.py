import torch
from torchmetrics.functional import accuracy, auroc, f1_score


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric_name = cfg.METRIC
    
    def __call__(self, logits, labels):
        if self.metric_name == "accuracy":
            return accuracy(logits, labels, task="multiclass", num_classes=self.cfg.NUM_CLASSES)
        elif self.metric_name == "auroc":
            probs = torch.softmax(logits, dim=1)[:, 1]
            return auroc(probs, labels, task="binary")
        elif self.metric_name == "f1_score":
            return f1_score(logits, labels, task="multilabel", num_labels=self.cfg.NUM_CLASSES, average="macro")
        else:
            raise ValueError(f"Metric {self.metric_name} not implemented.")
