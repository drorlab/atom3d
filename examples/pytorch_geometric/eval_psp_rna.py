import pandas as pd
import numpy as np
import sys

typ = sys.argv[1] # rna or psp

rps = []
rss = []
rks = []
pts = []
for seed in [251, 279, 947]:
    res_csv = f'../../data/{typ}/logs/test_{seed}/test_results.txt'
    res = pd.read_csv(res_csv)

    per_target = []
    for key, val in res.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])
    pts.append(per_target)
    
    

    true = res['true'].astype(float)
    pred = res['pred'].astype(float)
    pearson = true.corr(pred, method='pearson')
    kendall = true.corr(pred, method='kendall')
    spearman = true.corr(pred, method='spearman')
    rps.append(pearson)
    rss.append(spearman)
    rks.append(kendall)

per_target = pd.concat(pts)

print(f"per-target:\n \
    {per_target['pearson'].mean():.3f} / {per_target['pearson'].std():.3f}\n \
    {per_target['kendall'].mean():.3f} / {per_target['kendall'].std():.3f}\n \
    {per_target['spearman'].mean():.3f} / {per_target['spearman'].std():.3f}\n")

print(f'avg pearson {np.mean(rps)}, avg kendall {np.mean(rks)}, avg spearman {np.mean(rss)}')
print(f'stdev pearson {np.std(rps)}, stdev kendall {np.std(rks)}, stdev spearman {np.std(rss)}')