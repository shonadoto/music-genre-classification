import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(y_pred, y_true):
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    return float(accuracy_score(y_true_np, y_pred_np))


def calculate_f1_macro(y_pred, y_true):
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    return float(f1_score(y_true_np, y_pred_np, average="macro"))


def calculate_metrics(y_pred, y_true):
    return {
        "accuracy": calculate_accuracy(y_pred, y_true),
        "f1_macro": calculate_f1_macro(y_pred, y_true),
    }
