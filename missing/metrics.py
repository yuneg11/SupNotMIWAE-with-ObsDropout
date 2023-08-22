# Original implementation: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
#
# Copyright 2020 Max Horn
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AD CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.N


import sys
import numpy as np
import tensorflow as tf
from functools import partial
import sklearn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    precision_score
)

def mae(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1)))



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def mc_metric_wrapper(metric, **kwargs):
    """Wrap metric for multi class classification.

    If classifiction task is binary, select minority label as positive.
    Otherwise compute weighted average over classes.
    """
    def wrapped(y_true, y_score):
        if y_true.ndim == 1 and y_score.ndim == 2:
            # Multi class classification task where gt is given as int class
            # indicator. First need to convert to one hot label.
            n_classes = y_score.shape[-1]
            y_true = to_one_hot(y_true, n_classes)
        return metric(y_true, y_score, **kwargs)
    return wrapped


def accuracy(y_true, y_score):
    """Compute accuracy using one-hot representaitons."""
    if isinstance(y_true, list) and isinstance(y_score, list):
        # Online scenario
        if y_true[0].ndim == 2 and y_score[0].ndim == 2:
            # Flatten to single (very long prediction)
            y_true = np.concatenate(y_true, axis=0)
            y_score = np.concatenate(y_score, axis=0)
    if y_score.ndim == 3 and y_score.shape[-1] == 1:
        y_score = np.ravel(y_score)
        y_true = np.ravel(y_true).astype(int)
        y_score = np.around(y_score).astype(int)
    if y_true.ndim == 2 and y_true.shape[-1] != 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = np.round(y_true).astype(int)
    if y_score.ndim == 2 and y_score.shape[-1] != 1:
        y_score = np.argmax(y_score, axis=-1)
    if y_score.ndim == 2 and y_score.shape[-1] == 1:
        y_score = np.round(y_score).astype(int)
    return accuracy_score(y_true, np.round(y_score).astype(int))

def recall(y_true,y_score):
    y_score = np.round(y_score).astype(int)

    return recall_score(y_true,y_score)

def precision(y_true,y_score):
    y_score = np.round(y_score).astype(int)

    return precision_score(y_true,y_score)



def mgp_wrapper(fn):
    def wrapped(y_true, y_score):
        np.set_printoptions(threshold=sys.maxsize)
        if isinstance(y_true, list) and isinstance(y_score, list):
            # Online scenario
            if y_true[0].ndim == 2:
                # Flatten to single (very long prediction)
                y_true = np.concatenate(y_true, axis=0)
                y_score = np.concatenate(y_score, axis=0)
        assert y_true.size == y_score.size
        return fn(np.ravel(y_true), np.ravel(y_score))
    return wrapped


def calibration(y_true, y_score, sample_weight=None, norm='l2',
                      n_bins=10, strategy='uniform', pos_label=None,
                      reduce_bias=True):
    if isinstance(y_true, list) and isinstance(y_score, list):
        # Online scenario
        if y_true[0].ndim == 2 and y_score[0].ndim == 2:
            # Flatten to single (very long prediction)
            y_true = np.concatenate(y_true, axis=0)
            y_score = np.concatenate(y_score, axis=0)
    if y_score.ndim == 3 and y_score.shape[-1] == 1:
        y_score = np.ravel(y_score)
        y_true = np.ravel(y_true).astype(int)
        y_score = np.around(y_score).astype(int)
    if y_true.ndim == 2 and y_true.shape[-1] != 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = np.round(y_true).astype(int)
    if y_score.ndim == 2 and y_score.shape[-1] != 1:
        y_score = np.argmax(y_score, axis=-1)
    if y_score.ndim == 2 and y_score.shape[-1] == 1:
        y_score = np.round(y_score).astype(int)
    y_prob = y_score
    labels = np.unique(y_true)

    if pos_label is None:
        pos_label = y_true.max()
    if pos_label not in labels:
        raise ValueError("pos_label=%r is not a valid label: "
                         "%r" % (pos_label, labels))
    y_true = np.array(y_true == pos_label, int)

    norm_options = ('l1', 'l2', 'max')
    if norm not in norm_options:
        raise ValueError(f'norm has to be one of {norm_options}, got: {norm}.')

    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]
    if sample_weight is not None:
        sample_weight = sample_weight[remapping]
    else:
        sample_weight = np.ones(y_true.shape[0])

    n_bins = int(n_bins)
    if strategy == 'quantile':
        quantiles = np.percentile(y_prob, np.arange(0, 1, 1.0 / n_bins) * 100)
    elif strategy == 'uniform':
        quantiles = np.arange(0, 1, 1.0 / n_bins)
    else:
        raise ValueError(
            f"Invalid entry to 'strategy' input. Strategy must be either "
            f"'quantile' or 'uniform'. Got {strategy} instead."
        )

    threshold_indices = np.searchsorted(y_prob, quantiles).tolist()
    threshold_indices.append(y_true.shape[0])
    avg_pred_true = np.zeros(n_bins)
    bin_centroid = np.zeros(n_bins)
    delta_count = np.zeros(n_bins)
    debias = np.zeros(n_bins)

    loss = 0.
    count = float(sample_weight.sum())
    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]
        # ignore empty bins
        if i_end == i_start:
            continue
        delta_count[i] = float(sample_weight[i_start:i_end].sum())
        avg_pred_true[i] = (np.dot(y_true[i_start:i_end],
                                   sample_weight[i_start:i_end])
                            / delta_count[i])
        bin_centroid[i] = (np.dot(y_prob[i_start:i_end],
                                  sample_weight[i_start:i_end])
                           / delta_count[i])
        if norm == "l2" and reduce_bias:
            delta_debias = (
                avg_pred_true[i] * (avg_pred_true[i] - 1) * delta_count[i]
            )
            delta_debias /= (count * delta_count[i] - 1)
            debias[i] = delta_debias

    if norm == "max":
        loss = np.max(np.abs(avg_pred_true - bin_centroid))
    elif norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        loss = np.sum(delta_loss) / count
    elif norm == "l2":
        delta_loss = (avg_pred_true - bin_centroid)**2 * delta_count
        loss = np.sum(delta_loss) / count
        if reduce_bias:
            loss += np.sum(debias)
        loss = np.sqrt(max(loss, 0.))
    return loss


def calibration_curve(y_true, y_prob, *, normalize=False, n_bins=5, strategy="uniform"):

    y_true = sklearn.utils.validation.column_or_1d(y_true)
    y_prob = sklearn.utils.validation.column_or_1d(y_prob)
    sklearn.utils.validation.check_consistent_length(y_true, y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob has values outside [0, 1] and normalize is set to False."
        )

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. Provided labels %s." % labels
        )
    y_true = sklearn.preprocessing.label_binarize(y_true, classes=labels)[:, 0]

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece_score = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

    return ece_score

weight_auc = partial(roc_auc_score, average='micro')
auprc = mgp_wrapper(average_precision_score)
auroc = mgp_wrapper(roc_auc_score)
brier = mgp_wrapper(brier_score_loss)
ece = mgp_wrapper(calibration_curve)
logloss = mgp_wrapper(log_loss)
micro_auroc = mgp_wrapper(weight_auc)
# recall = mgp_wrapper(recall_score)


def balanced_wrapper(fun, **kwargs1):
    @mgp_wrapper
    def wrapper(y_true, y_prob, **kwargs):
        assert y_true.ndim == 1 and y_prob.ndim == 1

        neg_idxs = np.where(y_true == 0)[0]
        neg_metric = fun(np.zeros((len(neg_idxs),), dtype=int), y_prob[neg_idxs], **kwargs, **kwargs1)

        pos_idxs = np.where(y_true == 1)[0]
        pos_metric = fun(np.ones((len(pos_idxs),), dtype=int), y_prob[pos_idxs], **kwargs, **kwargs1)

        metric = (neg_metric + pos_metric) / 2
        return metric
    return wrapper



bal_brier = balanced_wrapper(brier_score_loss, pos_label=1)
bal_ece = balanced_wrapper(calibration_curve)
bal_logloss = balanced_wrapper(log_loss, labels=[0, 1])





# def flatten_inputs(func):
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs).flatten()
#     return wrapper
