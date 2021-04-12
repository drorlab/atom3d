import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

# Define the training run 
maxnumat='400'
name = 'lep_cutoff-08_maxnum-'+maxnumat+'-noh'
print(name)

# Load training results
rloader = res.ResultsENN(name, reps=[1,2,3])
results = rloader.get_all_predictions()

# Apply Softmax to outputs and use second [1] component to predict binary label
softmax = torch.nn.Softmax(dim=1)
for rep in results.values():
    for split, values in rep['predict'].items():
        score = softmax(torch.tensor(values))[:,1]
        rep['predict'][split] = score.numpy()

# Calculate and print results
summary = met.evaluate_average(results, metric = met.auroc, verbose = False)
print('AUROC: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.auprc, verbose = False)
print('AUPRC: %6.3f \pm %6.3f'%summary[2])

