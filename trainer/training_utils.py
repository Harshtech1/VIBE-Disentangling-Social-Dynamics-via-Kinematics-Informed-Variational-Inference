"""Headless training utilities shared by the trainer and CPU validation gates."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


CANONICAL_LABEL_ORDER = [0, 1, 2]
CANONICAL_CLASS_NAMES = ['Positive', 'Neutral', 'Negative']


def compute_evaluation_metrics(targets, preds):
    """Compute metrics in canonical order: Positive=0, Neutral=1, Negative=2.

    Macro scores weight all classes equally; weighted F1 weights each class by
    its support. One shared implementation prevents reporting-order drift.
    """
    acc = accuracy_score(targets, preds)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        targets, preds, labels=CANONICAL_LABEL_ORDER, average='macro', zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        targets, preds, labels=CANONICAL_LABEL_ORDER, average=None, zero_division=0
    )
    matrix = confusion_matrix(targets, preds, labels=CANONICAL_LABEL_ORDER)
    return {
        'acc': float(acc), 'f1': float(weighted_f1), 'weighted_f1': float(weighted_f1),
        'macro_f1': float(macro_f1), 'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'per_class': {
            name: {'precision': float(per_precision[index]), 'recall': float(per_recall[index]),
                   'f1': float(per_f1[index]), 'support': int(per_support[index])}
            for index, name in enumerate(CANONICAL_CLASS_NAMES)
        },
        'confusion_matrix': matrix.tolist(), 'class_order': CANONICAL_CLASS_NAMES,
    }


class WarmupCosineScheduler:
    """Epoch-indexed linear warmup followed by cosine annealing."""
    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs, eta_min=0.0):
        if total_epochs <= warmup_epochs:
            raise ValueError("total_epochs must exceed warmup_epochs")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = float(eta_min)
        self.epoch = 0

    def lr_for_epoch(self, epoch):
        if not 1 <= epoch <= self.total_epochs:
            raise ValueError(f"Epoch {epoch} outside 1..{self.total_epochs}")
        if epoch <= self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + np.cos(np.pi * progress))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        lr = self.lr_for_epoch(self.epoch)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self):
        return {"epoch": self.epoch, "base_lr": self.base_lr, "total_epochs": self.total_epochs,
                "warmup_epochs": self.warmup_epochs, "eta_min": self.eta_min}

    def load_state_dict(self, state):
        self.epoch = int(state["epoch"])
        if self.epoch:
            self.set_epoch(self.epoch)
