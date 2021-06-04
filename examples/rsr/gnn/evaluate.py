import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

# Define the training run 
name = 'logs/rsr_test/rsr'
print(name)

# Load training results
rloader = res.ResultsGNN(name, reps=[0,1,2])
results = rloader.get_target_specific_predictions()

# Calculate and print results
summary = met.evaluate_per_target_average(results['per_target'], metric = met.spearman, verbose = False)
print('Test Spearman (per-target): %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_per_target_average(results['per_target'], metric = met.pearson, verbose = False)
print('Test Pearson (per-target): %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_per_target_average(results['per_target'], metric = met.kendall, verbose = False)
print('Test Kendall (per-target): %6.3f \pm %6.3f'%summary[2])

summary = met.evaluate_average(results['global'], metric = met.spearman, verbose = False)
print('Test Spearman (global): %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results['global'], metric = met.pearson, verbose = False)
print('Test Pearson (global): %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results['global'], metric = met.kendall, verbose = False)
print('Test Kendall (global): %6.3f \pm %6.3f'%summary[2])

