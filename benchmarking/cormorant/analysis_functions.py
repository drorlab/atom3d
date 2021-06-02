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



def load_prediction_data(pr_file):
    
    pr_data = torch.load(pr_file)
    
    targets = np.array( pr_data['targets']*pr_data['sigma']+pr_data['mu'] )
    predict = np.array( pr_data['predict']*pr_data['sigma']+pr_data['mu'] )
    
    return targets, predict


def load_cormorant_log(log_fn):
    """
    Reads logged loss metrics of each epoch from a Cormorant log file.
       - for regression: MAE and RMSE 
       - for classification: crossent and accuracy
    """
    mae_tr, rmse_tr = [], [] # training loss
    mae_va, rmse_va = [], [] # validation loss
    with open(log_fn,'r') as in_file:
        while True: 
            # Read the next line 
            line = in_file.readline() 
            if not line: 
                break
            # Read the loss
            if 'Current Training Loss' in line:
                columns = line.split()
                mae_tr.append(float(columns[9]))
                rmse_tr.append(float(columns[10]))
            if 'Current Validation Loss' in line:
                columns = line.split()
                mae_va.append(float(columns[9]))
                rmse_va.append(float(columns[10]))
    # convert to numpy arrays
    mae_tr, rmse_tr = np.array(mae_tr), np.array(rmse_tr)
    mae_va, rmse_va = np.array(mae_va), np.array(rmse_va)
    return mae_tr, rmse_tr, mae_va, rmse_va


def plot_cormorant_log(log_fn):
    
    # Load the Cormorant log
    mae_tr, rmse_tr, mae_va, rmse_va = load_cormorant_log(log_fn)
    
    # Create the figure
    fig, ax = plt.subplots(2, 1, figsize=[4,4], sharex=True, dpi=100)
    # Plot data
    epoch = np.arange(1, len(mae_tr)+1)
    ax[0].plot(epoch, mae_tr, label='training')
    ax[1].plot(epoch, rmse_tr, label='training') 
    ax[0].plot(epoch, mae_va, label='validation')
    ax[1].plot(epoch, rmse_va, label='validation')
    # Limits
    ax[0].set_xlim(0, len(mae_tr)+1)
    #ax[0].set_ylim(0, np.max(mae_tr)+0.05)
    #ax[1].set_ylim(0, np.max(rmse_tr)+0.05)
    # Labels
    ax[0].set_ylabel('MAE')
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('epoch')
    ax[0].legend()
    ax[1].legend()
    # Layout
    fig.tight_layout()
    

def average_r2(name, reps=[1,2,3], verbose=True):
    """
    Calculate the average R^2 for training and test data.
    """
    # Initialization
    r2_tr = np.empty(len(reps))
    r2_va = np.empty(len(reps))
    r2_te = np.empty(len(reps))
    if verbose: print('Pearson R^2')
    for r, rep in enumerate(reps):
        prediction_fn = name + '-rep'+str(int(rep))+'.best'
        # Load the Cormorant predictions
        targets_tr, predict_tr = load_prediction_data(prediction_fn+'.train.pt')
        targets_va, predict_va = load_prediction_data(prediction_fn+'.valid.pt')
        targets_te, predict_te = load_prediction_data(prediction_fn+'.test.pt')
        # Calculate Statistics
        r2_tr[r] = stats.pearsonr(targets_tr, predict_tr)[0]**2
        r2_va[r] = stats.pearsonr(targets_va, predict_va)[0]**2
        r2_te[r] = stats.pearsonr(targets_te, predict_te)[0]**2
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, r2_tr[r], r2_va[r], r2_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(r2_tr), np.std(r2_tr), np.mean(r2_va), np.std(r2_va), np.mean(r2_te), np.std(r2_te)


def average_mae(name, reps=[1,2,3], verbose=True):
    """
    Calculate the average MAE for training and test data.
    """
    # Initialization
    mae_tr = np.empty(len(reps))
    mae_va = np.empty(len(reps))
    mae_te = np.empty(len(reps))
    if verbose: print('Mean Absolute Error')
    for r, rep in enumerate(reps):
        prediction_fn = name + '-rep'+str(int(rep))+'.best'
        # Load the Cormorant predictions
        targets_tr, predict_tr = load_prediction_data(prediction_fn+'.train.pt')
        targets_va, predict_va = load_prediction_data(prediction_fn+'.valid.pt')
        targets_te, predict_te = load_prediction_data(prediction_fn+'.test.pt')
        # Calculate Statistics
        mae_tr[r] = np.mean(np.abs(targets_tr - predict_tr))
        mae_te[r] = np.mean(np.abs(targets_te - predict_te))
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, mae_tr[r], mae_va[r], mae_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(mae_tr), np.std(mae_tr), np.mean(mae_va), np.std(mae_va), np.mean(mae_te), np.std(mae_te)


def average_rmse(name, reps=[1,2,3], verbose=True):
    """
    Calculate the average RMSE for training and test data.
    """
    # Initialization
    rmse_tr = np.empty(len(reps))
    rmse_va = np.empty(len(reps))
    rmse_te = np.empty(len(reps))
    if verbose: print('Root Mean Squared Error')
    for r, rep in enumerate(reps):
        prediction_fn = name + '-rep'+str(int(rep))+'.best'
        # Load the Cormorant predictions
        targets_tr, predict_tr = load_prediction_data(prediction_fn+'.train.pt')
        targets_va, predict_va = load_prediction_data(prediction_fn+'.valid.pt')
        targets_te, predict_te = load_prediction_data(prediction_fn+'.test.pt')
        # Calculate Statistics
        rmse_tr[r] = np.sqrt(((predict_tr - targets_tr) ** 2).mean())
        rmse_va[r] = np.sqrt(((predict_va - targets_va) ** 2).mean())
        rmse_te[r] = np.sqrt(((predict_te - targets_te) ** 2).mean())
        if verbose: print(' - Rep.: %i - Training: %5.3f - Validation: %5.3f - Test: %5.3f'%(rep, rmse_tr[r], rmse_va[r], rmse_te[r]))
    # Mean and corresponding standard deviations
    return np.mean(rmse_tr), np.std(rmse_tr), np.mean(rmse_va), np.std(rmse_va), np.mean(rmse_te), np.std(rmse_te)

