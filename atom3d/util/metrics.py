import os, sys
import pickle
import numpy as np
import scipy as sp
import scipy.stats as stats
import sklearn.metrics

    
def pearson(targets, predict):
    r = stats.pearsonr(targets, predict)[0]
    return r
    
def spearman(targets, predict):
    rho = stats.spearmanr(targets, predict)[0]
    return rho
    
def kendall(targets, predict):
    tau = stats.kendalltau(targets, predict)[0]
    return tau

def r2(targets, predict):
    r = pearson(targets, predict)
    return r**2
        
def mae(targets, predict):
    abs_err = np.abs(targets - predict)
    return np.mean(abs_err)

def rmse(targets, predict):
    sq_err = ((predict - targets) ** 2)
    return np.sqrt(np.mean(sq_err))
    
def auroc(targets, predict):
    auc = sklearn.metrics.roc_auc_score(targets, predict)
    return auc
    
def auprc(targets, predict):
    ap = sklearn.metrics.average_precision_score(targets, predict)
    return ap

def accuracy(targets, predict):
    ac = sklearn.metrics.accuracy_score(targets, predict)
    return ac
    
    
def evaluate_average(results, metric=r2, verbose=True):
    """
    Calculate metric for training, validation and test data, averaged over all replicates.
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

