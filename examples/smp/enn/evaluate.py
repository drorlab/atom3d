import numpy as np
import atom3d.util.results as res
import atom3d.util.metrics as met

labels = np.loadtxt('labels.txt',dtype=str)
conversion = [1., 1., 1., 1., 1., 
              27.2114, 27.2114, 27.2114,
              1., 27211.4, 1., 1., 1., 1., 1., 
              23.061, 23.061, 23.061, 23.061]

for cf, label in zip(conversion,labels):
    rloader = res.ResultsENN('smp-'+label, reps=[1,2,3])
    results = rloader.get_all_predictions()
    summary = met.evaluate_average(results, metric = met.mae, verbose = False)
    summary = [(cf*s[0],cf*s[1]) for s in summary]
    print('%9s: %6.3f \pm %6.3f'%(label, *summary[2]))