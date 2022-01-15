import sys
import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

seqid = sys.argv[1]

# Define the training run 
name = f'logs/lba_test_withH_{seqid}/lba'
print(name)

# Load training results
rloader = res.ResultsGNN(name, reps=[0,1,2])
results = rloader.get_all_predictions()

# Calculate and print results
summary = met.evaluate_average(results, metric = met.rmse, verbose = False)
print('Test RMSE: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.spearman, verbose = False)
print('Test Spearman: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.pearson, verbose = False)
print('Test Pearson: %6.3f \pm %6.3f'%summary[2])

