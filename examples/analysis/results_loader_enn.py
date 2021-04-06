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



class LoaderENN():

    def __init__(self, name, reps=[1,2,3]):
        self.name = name
        self.reps = reps

    def get_prediction(self, prediction_fn):
        """
        Reads targets and prediction from pt file.
        """
        pr_data = torch.load(prediction_fn)
        targets = np.array( pr_data['targets']*pr_data['sigma']+pr_data['mu'] )
        predict = np.array( pr_data['predict']*pr_data['sigma']+pr_data['mu'] )
        return targets, predict

    def get_all_predictions(self):
        results = {}
        for r, rep in enumerate(self.reps):
            prediction_fn = self.name + '-rep'+str(int(rep))+'.best'
            # Load the Cormorant predictions
            targets_tr, predict_tr = self.get_prediction(prediction_fn+'.train.pt')
            targets_va, predict_va = self.get_prediction(prediction_fn+'.valid.pt')
            targets_te, predict_te = self.get_prediction(prediction_fn+'.test.pt')
            targets = {'train':targets_tr, 'valid':targets_va, 'test':targets_te}
            predict = {'train':predict_tr, 'valid':predict_va, 'test':predict_te}
            results['rep'+str(int(rep))] = {'targets':targets, 'predict':predict}
        return results

    def get_log(self, log_file):
        """
        Reads logged loss metrics of each epoch from a Cormorant log file.
         - for regression: MAE and RMSE 
         - for classification: crossent and accuracy
        """
        mae_tr, rmse_tr = [], [] # training loss
        mae_va, rmse_va = [], [] # validation loss
        with open(log_file,'r') as in_file:
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

    def plot_log(self, rep):
    
        # Load the Cormorant log
        mae_tr, rmse_tr, mae_va, rmse_va = self.get_log(self.name+'-'+rep+'.log')
    
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

