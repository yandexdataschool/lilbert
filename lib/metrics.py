import numpy as np
from sklearn import metrics


def accuracy(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.accuracy_score(true_labels, pred_labels)


def log_loss(true_labels, prob_preds):
    return metrics.log_loss(true_labels, prob_preds)


def f1_score(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.f1_score(true_labels, pred_labels)


def matthews_corrcoef(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.matthews_corrcoef(true_labels, pred_labels)
    