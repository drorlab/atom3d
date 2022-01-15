import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

# Define the training run 
name = 'logs/msp_test/msp'
print(name)

# Load training results
rloader = res.ResultsGNN(name, reps=[0,1,2])
results = rloader.get_all_predictions()

y_true = results['rep0']['targets']['test']
y_pred = results['rep0']['predict']['test']

np.save('/oak/stanford/groups/rbaltman/aderry/COLLAPSE/results/msp_atom3d_y_true', y_true)
np.save('/oak/stanford/groups/rbaltman/aderry/COLLAPSE/results/msp_atom3d_y_pred', y_pred)

# Calculate and print results
summary = met.evaluate_average(results, metric = met.auroc, verbose = False)
print('Test AUROC: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.auprc, verbose = False)
print('Test AUPRC: %6.3f \pm %6.3f'%summary[2])

