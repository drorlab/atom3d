# Standard modules
import os, sys
import pickle
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

# Pytorch for data set
import torch
from torch.utils.data import Dataset, DataLoader


def average_r2(results, verbose=True):
    """
    Calculate the average R^2 for training and test data.
    """
    # Initialization
    reps = results.keys()
    r2_tr = np.empty(len(reps))
    r2_va = np.empty(len(reps))
    r2_te = np.empty(len(reps))
    if verbose: print('Pearson R^2')
    for r, rep in enumerate(results.keys()):
        # Load the predictions
        targets_tr, predict_tr = results[rep]['targets']['train'], results[rep]['predict']['train']
        targets_va, predict_va = results[rep]['targets']['valid'], results[rep]['predict']['valid']
        targets_te, predict_te = results[rep]['targets']['test'],  results[rep]['predict']['test']
        # Calculate Statistics
        r2_tr[r] = stats.pearsonr(targets_tr, predict_tr)[0]**2
        r2_va[r] = stats.pearsonr(targets_va, predict_va)[0]**2
        r2_te[r] = stats.pearsonr(targets_te, predict_te)[0]**2
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, r2_tr[r], r2_va[r], r2_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(r2_tr), np.std(r2_tr), np.mean(r2_va), np.std(r2_va), np.mean(r2_te), np.std(r2_te)


def average_mae(results, verbose=True):
    """
    Calculate the average MAE for training and test data.
    """
    # Initialization
    reps = results.keys()
    mae_tr = np.empty(len(reps))
    mae_va = np.empty(len(reps))
    mae_te = np.empty(len(reps))
    if verbose: print('Mean Absolute Error')
    for r, rep in enumerate(reps):
        # Load the predictions
        targets_tr, predict_tr = results[rep]['targets']['train'], results[rep]['predict']['train']
        targets_va, predict_va = results[rep]['targets']['valid'], results[rep]['predict']['valid']
        targets_te, predict_te = results[rep]['targets']['test'],  results[rep]['predict']['test']
        # Calculate Statistics
        mae_tr[r] = np.mean(np.abs(targets_tr - predict_tr))
        mae_te[r] = np.mean(np.abs(targets_te - predict_te))
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, mae_tr[r], mae_va[r], mae_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(mae_tr), np.std(mae_tr), np.mean(mae_va), np.std(mae_va), np.mean(mae_te), np.std(mae_te)


def average_rmse(results, verbose=True):
    """
    Calculate the average RMSE for training and test data.
    """
    # Initialization
    reps = results.keys()
    rmse_tr = np.empty(len(reps))
    rmse_va = np.empty(len(reps))
    rmse_te = np.empty(len(reps))
    if verbose: print('Root Mean Squared Error')
    for r, rep in enumerate(reps):
        # Load the predictions
        targets_tr, predict_tr = results[rep]['targets']['train'], results[rep]['predict']['train']
        targets_va, predict_va = results[rep]['targets']['valid'], results[rep]['predict']['valid']
        targets_te, predict_te = results[rep]['targets']['test'],  results[rep]['predict']['test']
        # Calculate Statistics
        rmse_tr[r] = np.sqrt(((predict_tr - targets_tr) ** 2).mean())
        rmse_va[r] = np.sqrt(((predict_va - targets_va) ** 2).mean())
        rmse_te[r] = np.sqrt(((predict_te - targets_te) ** 2).mean())
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, rmse_tr[r], rmse_va[r], rmse_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(rmse_tr), np.std(rmse_tr), np.mean(rmse_va), np.std(rmse_va), np.mean(rmse_te), np.std(rmse_te)

