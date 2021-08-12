import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met

for name in ['res_maxnum-400_samples-100']:
    print(name)
    rloader = res.ResultsENN(name, reps=[1,2,3])
    results = rloader.get_all_predictions()
    # Apply Softmax to outputs and use second [1] component to predict binary label
    #softmax = torch.nn.Softmax(dim=1)
    #print(softmax)
    for rep in results.values():
        for split, values in rep['predict'].items():
            print('Values '+split+':')
            print(values)
            print('Dimensions:',values.shape)
            #score = softmax(torch.tensor(values))[:,1]
            #rep['predict'][split] = score.numpy()
            rep['predict'][split] = np.argmax(values, axis=1)
    # Calculate and print results
    summary = met.evaluate_average(results, metric = met.accuracy, verbose = False)
    print('accuracy: %6.3f $\pm$ %6.3f'%summary[2])

