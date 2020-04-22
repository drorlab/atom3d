import numpy as np
import pandas as pd
import json
import os
import argparse
from data_util import *
from sklearn.model_selection import KFold

def check_identity(train, test):
    """
    takes two lists corresponding to sets of proteins. Each item is a list with 1 or more clusters
    """
    invalid = []
    test_clusts = set([c for t in test for c in t])
    for i, samp in enumerate(train):
        for cl in samp:
            if cl in test_clusts:
                invalid.append(i)
                break
    print('removing', len(invalid), 'out of', len(train), 'examples')
    valid_idx = [i for i,t in enumerate(train) if i not in invalid]
    return valid_idx

def identity_filter(train, test, val=None):
    if val:
        print('filtering train set...')
        train_to_keep = check_identity(train.cluster, val.cluster)
        train = train.iloc[train_to_keep,:]
    print('filtering validation set...')
    train_to_keep = check_identity(train.cluster, test.cluster)
    train = train.iloc[train_to_keep,:]
    return train, test, val

def map_clusters(data):
    data = data.reset_index()
    clusters = {}
    for i in range(data.shape[0]):
        for c in data.loc[i, 'cluster']:
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(data.loc[i, 'pdb'])
    return clusters

def combine_clusters(clusters):
    combined_clusters = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
    curr = 9
    for cluster, pdbs in clusters.items():
        if len(pdbs) == 1:
            combined_clusters[1].extend(pdbs)
        elif len(pdbs) <= 2:
            combined_clusters[2].extend(pdbs)
        elif len(pdbs) <= 4:
            combined_clusters[3].extend(pdbs)
        elif len(pdbs) <= 6:
            combined_clusters[4].extend(pdbs)
        elif len(pdbs) <= 10:
            combined_clusters[5].extend(pdbs)
        elif len(pdbs) <= 20:
            combined_clusters[6].extend(pdbs)
        elif len(pdbs) <= 50:
            combined_clusters[7].extend(pdbs)
        elif len(pdbs) <= 200:
            combined_clusters[8].extend(pdbs)
        else:
            combined_clusters[curr] = pdbs
            curr += 1
    return combined_clusters

def cluster_cv(data, clusters):
    cv_splits = []
    for pdbs in clusters.values():
        train = data.pdb[[p not in pdbs for p in data.pdb]]
        val = data.pdb[[p in pdbs for p in data.pdb]]
        cv_splits.append((train, val))
    return cv_splits


def time_split(data, identity=None, write_to_file=False):
    val = data.query('year == "2016" | year == "2017"')
    test = data.query('year == "2018"')
    train = data[[p not in val.pdb.tolist() + test.pdb.tolist() for p in data.pdb]]
    
    if identity:
        train, test, val = identity_filter(train, test)
    
    train_pdbs = train['pdb'].tolist()
    val_pdbs = val['pdb'].tolist()
    test_pdbs = test['pdb'].tolist()
    
    print(f'{len(train_pdbs)} train, {len(val_pdbs)} val, {len(test_pdbs)} test examples')
    
    if write_to_file:
        if identity:
            train['pdb'].to_csv(f'splits/time_split/train_{identity}.txt', index=False, header=False)
            val['pdb'].to_csv(f'splits/time_split/val_{identity}.txt', index=False, header=False)
            test['pdb'].to_csv(f'splits/time_split/test_{identity}.txt', index=False, header=False)
        else:
            train['pdb'].to_csv(f'splits/time_split/train.txt', index=False, header=False)
            val['pdb'].to_csv(f'splits/time_split/val.txt', index=False, header=False)
            test['pdb'].to_csv(f'splits/time_split/test.txt', index=False, header=False)
    
    return train_pdbs, val_pdbs, test_pdbs


def core_split(data, core_pdbs, cv_method, write_to_file=False):
    test = data[[p in core_pdbs for p in data.pdb]]
    train = data[[p not in core_pdbs for p in data.pdb]]
    print(f'{len(train)} train, {len(test)} test examples')
    
    if cv_method == 'random':
        print('splitting into 10 folds randomly...')
        cv_splits = []
        cv = KFold(n_splits=10, shuffle=True, random_state=20)
        for train_idx, val_idx in cv.split(train):
            train_pdbs = train.iloc[train_idx,:].pdb
            val_pdbs = train.iloc[val_idx,:].pdb
            cv_splits.append((train_pdbs,val_pdbs))
    
    elif cv_method == 'cluster':
        print('splitting into 10 folds by cluster...')
        clusters = map_clusters(train)
        combined_clusters = combine_clusters(clusters)
        cv_splits = cluster_cv(train, combined_clusters)
    
    if write_to_file:
        for i, (train_pdbs, val_pdbs) in enumerate(cv_splits):
            train_pdbs.to_csv(f'splits/core_split/train_{cv_method}_cv{i}.txt', index=False, header=False)
            val_pdbs.to_csv(f'splits/core_split/val_{cv_method}_cv{i}.txt', index=False, header=False)
        test.pdb.to_csv(f'splits/core_split/test.txt', index=False, header=False)
    
    return cv_splits

def cluster_split(data, write_to_file=False):
    pass


def main(split, datapath, cv_method, identity, write_to_file):
    refined_set = pd.read_csv(os.path.join(datapath, 'pdbbind_refined_set_cleaned.csv'))
    
    if identity:
        pdb2cluster, _ = get_pdb_clusters(identity)
        refined_set['cluster'] = refined_set.pdb.apply(lambda x: pdb2cluster[x])
    
    if split == 'time':
        train_pdbs, val_pdbs, test_pdbs = time_split(refined_set, identity, write_to_file)
    
    elif split == 'core':
        with open(os.path.join(datapath, 'core_pdbs.txt')) as f:
            core_pdbs = f.read().splitlines()
        cv_splits = core_split(refined_set, core_pdbs, cv_method, write_to_file)
    
    elif split == 'cluster':
        cv_splits = cluster_split(refined_set, write_to_file)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=str, help='split type. can be "time", "core"')
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    parser.add_argument('--cv_method', type=str, default='random', help='cross-validation strategy, either "random" or "clustered". Default: random')
    parser.add_argument('--identity', type=int, default=None, help='sequence identity cutoff for splitting. Must be used with --cv_method=cluster Default: none')
    parser.add_argument('--write_to_file', action='store_true', help='write the split pdbs to files (T/F)')
    args = parser.parse_args()
    if args.identity and args.identity not in [30,40,50,70,90,95,100]:
        raise Exception('invalid identity cutoff. possible values = 30,40,50,70,90,95,100')
    
    main(**vars(args))