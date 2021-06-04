import numpy as np
import atom3d.util.results as res
import atom3d.util.metrics as met

labels = np.loadtxt('labels.txt', dtype=str)
conversion = {'A':1.0, 'B':1.0, 'C':1.0, 'mu':1.0, 'alpha':1.0,
              'homo':27.2114, 'lumo':27.2114, 'gap':27.2114, 'r2':1.0, 'zpve':27211.4,
              'u0':27.2114, 'u298':27.2114, 'h298':27.2114, 'g298':27.2114, 'cv':1.0,
              'u0_atom':27.2114, 'u298_atom':27.2114, 'h298_atom':27.2114, 'g298_atom':27.2114, 'cv_atom':1.0}

for label in labels:
    name = f'logs/smp_test_{label}/smp'
    cf = conversion[label]
    rloader = res.ResultsGNN(name, reps=[0,1,2])
    results = rloader.get_all_predictions()
    summary = met.evaluate_average(results, metric = met.mae, verbose = False)
    summary = [(cf*s[0],cf*s[1]) for s in summary]
    print('%9s: %6.3f \pm %6.3f'%(label, *summary[2]))

