import os, sys
import pickle
import torch
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats


class Results3DCNN():

    def __init__(self, name, reps=[1,2,3]):
        self.name = name
        self.reps = reps

    def get_prediction(self, prediction_fn):
        """
        Reads targets and prediction.
        
        TODO: Implement this!
        
        """
        targets, predict = None, None
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
        

class ResultsGNN():

    def __init__(self, name, reps=[1,2,3]):
        self.name = name
        self.reps = reps

    def get_prediction(self, prediction_fn):
        """
        Reads targets and prediction
        """
        pr_data = torch.load(prediction_fn)
        targets = np.array( pr_data['targets'] )
        predict = np.array( pr_data['predictions'] )
        return targets, predict

    def get_all_predictions(self):
        results = {}
        for r, rep in enumerate(self.reps):
            prediction_fn = self.name + '-rep'+str(int(rep))+'.best'
            targets_tr, predict_tr = self.get_prediction(prediction_fn+'.train.pt')
            targets_va, predict_va = self.get_prediction(prediction_fn+'.val.pt')
            targets_te, predict_te = self.get_prediction(prediction_fn+'.test.pt')
            targets = {'train':targets_tr, 'valid':targets_va, 'test':targets_te}
            predict = {'train':predict_tr, 'valid':predict_va, 'test':predict_te}
            results['rep'+str(int(rep))] = {'targets':targets, 'predict':predict}
        return results
    
    def get_predictions_by_target(self, prediction_fn):
        results_df = pd.DataFrame(torch.load(prediction_fn))
        per_target = []
        for key, val in results_df.groupby(['target']):
            # Ignore target with 2 decoys only since the correlations are
            # not really meaningful.
            if val.shape[0] < 3:
                continue
            true = val['true'].astype(float).to_numpy()
            pred = val['pred'].astype(float).to_numpy()
            per_target.append((true, pred))
        global_true = results_df['true'].astype(float).to_numpy()
        global_pred = results_df['pred'].astype(float).to_numpy()
        return global_true, global_pred, per_target
    
    def get_target_specific_predictions(self):
        """For use with PSR/RSR. Here `target` refers to the protein target, not the prediction target."""
        results = {'global':{}, 'per_target':{}}
        for r, rep in enumerate(self.reps):
            prediction_fn = self.name + '-rep'+str(int(rep))+'.best'
            targets_tr, predict_tr, per_target_tr = self.get_predictions_by_target(prediction_fn+'.train.pt')
            targets_va, predict_va, per_target_va = self.get_predictions_by_target(prediction_fn+'.val.pt')
            targets_te, predict_te, per_target_te = self.get_predictions_by_target(prediction_fn+'.test.pt')
            targets = {'train':targets_tr, 'valid':targets_va, 'test':targets_te}
            predict = {'train':predict_tr, 'valid':predict_va, 'test':predict_te}
            per_target = {'train': per_target_tr, 'valid':per_target_va, 'test':per_target_te}
            results['global']['rep'+str(int(rep))] = {'targets':targets, 'predict':predict}
            results['per_target']['rep'+str(int(rep))] = per_target
        return results
            

class ResultsENN():

    def __init__(self, name, reps=[1,2,3]):
        self.name = name
        self.reps = reps

    def get_prediction(self, prediction_fn):
        """
        Reads targets and prediction from pt file.
        """
        pr_data = torch.load(prediction_fn)
        if 'sigma' in pr_data.keys():
            targets = np.array( pr_data['targets']*pr_data['sigma']+pr_data['mu'] )
            predict = np.array( pr_data['predict']*pr_data['sigma']+pr_data['mu'] )
        else:
            targets = np.array( pr_data['targets'] )
            predict = np.array( pr_data['predict'] )
        return targets, predict

    def get_all_predictions(self):
        results = {}
        for r, rep in enumerate(self.reps):
            prediction_fn = 'predict/' + self.name + '-rep'+str(int(rep))+'.best'
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

