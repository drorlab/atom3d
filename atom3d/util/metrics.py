import os, sys
import pickle
import numpy as np
import scipy as sp
import scipy.stats as stats
import sklearn.metrics

    
def pearson(targets, predict):
    """
    Calculate the Pearson correlation coefficient (R) between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: Pearson correlation coefficient.
    :rtype: float

    """
    r = stats.pearsonr(targets, predict)[0]
    return r
    
def spearman(targets, predict):
    """
    Calculate the Spearman correlation coefficient (rho) between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: Spearman correlation coefficient.
    :rtype: float

    """
    rho = stats.spearmanr(targets, predict)[0]
    return rho
    
def kendall(targets, predict):
    """
    Calculate the Kendall tau (a correlation measure for for ordinal data) between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: The tau statistic.
    :rtype: float

    """
    tau = stats.kendalltau(targets, predict)[0]
    return tau

def r2(targets, predict):
    """
    Calculate R^2, the square of the Pearson correlation coefficient, between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: Square of the Pearson correlation coefficient.
    :rtype: float

    """
    r = pearson(targets, predict)
    return r**2
        
def mae(targets, predict):
    """
    Calculate the mean absolute error between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: The mean absolute error.
    :rtype: float

    """
    abs_err = np.abs(targets - predict)
    return np.mean(abs_err)

def rmse(targets, predict):
    """
    Calculate the root mean squared error between targets and prediction.

    :param targets: The target values (ground truth).
    :type targets: numpy.ndarray
    :param predict: The predictions.
    :type predict: numpy.ndarray
    
    :return: The root mean squared error.
    :rtype: float

    """
    sq_err = ((predict - targets) ** 2)
    return np.sqrt(np.mean(sq_err))
    
def auroc(targets, predict):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    
    :param targets: The target scores (ground truth).
    :type targets: numpy.ndarray
    :param predict: The prediction scores.
    :type predict: numpy.ndarray
    
    :return: ROC AUC.
    :rtype: float    
    
    """
    auc = sklearn.metrics.roc_auc_score(targets, predict)
    return auc
    
def auprc(targets, predict):
    """
    Compute Area Under the Precision-Recall Curve (PRC AUC) from prediction scores.
    This ia also called the average precision (AP).
    
    :param targets: The target scores (ground truth).
    :type targets: numpy.ndarray
    :param predict: The prediction scores.
    :type predict: numpy.ndarray
    
    :return: The average precision.
    :rtype: float    
    
    """
    ap = sklearn.metrics.average_precision_score(targets, predict)
    return ap

def accuracy(targets, predict):
    """
    Compute the accuracy from prediction scores.
    
    :param targets: The target scores (ground truth).
    :type targets: numpy.ndarray
    :param predict: The prediction scores.
    :type predict: numpy.ndarray
    
    :return: The accuracy.
    :rtype: float    
    
    """
    ac = sklearn.metrics.accuracy_score(targets, predict)
    return ac
    
    
def evaluate_average(results, metric=r2, verbose=True, select=None):
    """
    Calculate a metric for training, validation and test data, averaged over all replicates.
    """
    # Initialization
    reps = results.keys()
    metric_tr = np.empty(len(reps))
    metric_va = np.empty(len(reps))
    metric_te = np.empty(len(reps))
    # Go through training repetitions
    for r, rep in enumerate(results.keys()):
        # Load the predictions
        targets_tr, predict_tr = results[rep]['targets']['train'], results[rep]['predict']['train']
        targets_va, predict_va = results[rep]['targets']['valid'], results[rep]['predict']['valid']
        targets_te, predict_te = results[rep]['targets']['test'],  results[rep]['predict']['test']
        # Calculate Statistics
        metric_tr[r] = metric(targets_tr, predict_tr)
        metric_va[r] = metric(targets_va, predict_va)
        metric_te[r] = metric(targets_te, predict_te)
        if verbose: print(' - %s  -  Training: %7.3f  -  Validation: %7.3f  -  Test: %7.3f'%(rep, metric_tr[r], metric_va[r], metric_te[r]))
    # Optionally select training runs with lowest/highest validation metrics
    if select is not None:
        order = np.argsort(metric_va)
        if select<0: 
            order = order[select:]
        else:
            order = order[:select]
        metric_tr = metric_tr[order]
        metric_va = metric_va[order]
        metric_te = metric_te[order]
    # Mean and corresponding standard deviations
    summary_tr = (np.mean(metric_tr), np.std(metric_tr))
    summary_va = (np.mean(metric_va), np.std(metric_va))
    summary_te = (np.mean(metric_te), np.std(metric_te))
    if verbose: 
        print('---')
        print(' Training:   %7.3f +/- %7.3f'%summary_tr)
        print(' Validation: %7.3f +/- %7.3f'%summary_va)
        print(' Test:       %7.3f +/- %7.3f'%summary_te)
    return summary_tr, summary_va, summary_te

def _per_target_mean(res, metric):
    all_res = []
    for targets, predictions in res:
        all_res.append(metric(targets, predictions))
    return np.mean(all_res)

def evaluate_per_target_average(results, metric=r2, verbose=True):
    """
    Calculate a metric for training, validation and test data, averaged over all replicates.
    """
    # Initialization
    reps = results.keys()
    metric_tr = np.empty(len(reps))
    metric_va = np.empty(len(reps))
    metric_te = np.empty(len(reps))
    # Go through training repetitions
    for r, rep in enumerate(results.keys()):
        # Load the predictions
        train = results[rep]['train']
        val = results[rep]['valid']
        test = results[rep]['test']
            
        # Calculate Statistics
        metric_tr[r] = _per_target_mean(train, metric)
        metric_va[r] = _per_target_mean(val, metric)
        metric_te[r] = _per_target_mean(test, metric)
        
        if verbose: print(' - %s  -  Training: %7.3f  -  Validation: %7.3f  -  Test: %7.3f'%(rep, metric_tr[r], metric_va[r], metric_te[r]))
    # Mean and corresponding standard deviations
    summary_tr = (np.mean(metric_tr), np.std(metric_tr))
    summary_va = (np.mean(metric_va), np.std(metric_va))
    summary_te = (np.mean(metric_te), np.std(metric_te))
    if verbose: 
        print('---')
        print(' Training:   %7.3f +/- %7.3f'%summary_tr)
        print(' Validation: %7.3f +/- %7.3f'%summary_va)
        print(' Test:       %7.3f +/- %7.3f'%summary_te)
    return summary_tr, summary_va, summary_te
