import numpy as np
import atom3d.util.results as res
import atom3d.util.metrics as met

cutoff='06'
maxnumat='600'

for id_split in ['30','60']:
    name = 'lba-id'+id_split+'-siamese_cutoff-'+cutoff+'_maxnumat-'+maxnumat
    print(name)
    rloader = res.ResultsENN(name, reps=[1,2,3])
    results = rloader.get_all_predictions()
    summary = met.evaluate_average(results, metric = met.rmse, verbose = False)
    print('RMSE: %6.3f \pm %6.3f'%summary[2])
    summary = met.evaluate_average(results, metric = met.pearson, verbose = False)
    print('R_P:  %6.3f \pm %6.3f'%summary[2])
    summary = met.evaluate_average(results, metric = met.spearman, verbose = False)
    print('R_S:  %6.3f \pm %6.3f'%summary[2])

