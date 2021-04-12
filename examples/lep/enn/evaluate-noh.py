import numpy as np
import atom3d.util.results as res
import atom3d.util.metrics as met

cutoff='08'
maxnumat='400'

name = 'lep_cutoff-'+cutoff+'_maxnum-'+maxnumat+'-noh'
print(name)
rloader = res.ResultsENN(name, reps=[1,2,3])
results = rloader.get_all_predictions()
summary = met.evaluate_average(results, metric = met.auroc, verbose = False)
print('AUROC: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.auprc, verbose = False)
print('AUPRC: %6.3f \pm %6.3f'%summary[2])

