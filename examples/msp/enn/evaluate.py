import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

for name in ['msp_cutoff-06_bs-4_LMDB', 'msp_cutoff-08_bs-4_LMDB-noH']:
    print(name)
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
    print('AUROC: %6.3f $\pm$ %6.3f'%summary[2])
    summary = met.evaluate_average(results, metric = met.auprc, verbose = False)
    print('AUPRC: %6.3f $\pm$ %6.3f'%summary[2])

