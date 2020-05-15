import argparse
import functools
import os
import tqdm

import numpy as np
import pandas as pd

import atom3d.psp.util as util
import atom3d.util.datatypes as dt
import atom3d.util.ensemble as en
import atom3d.util.file as fi
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho
import atom3d.util.splits as sp


def split_targets_random(targets_df, train_size=None, val_size=0.1,
                         test_size=0.1, shuffle=True, random_seed=None):
    """
    Randomly split targets for train/val/test.
    """
    test_indices, val_indices, train_indices = sp.random_split(
        len(targets_df), train_split=train_size, vali_split=val_size,
        test_split=test_size, shuffle=shuffle, random_seed=random_seed)

    all_targets = targets_df.target.values
    targets_train = all_targets[train_indices]
    targets_val = all_targets[val_indices]
    targets_test = all_targets[val_indices]
    return targets_train, targets_val, targets_test


def split_targets_by_year(targets_df, test_years, train_years=None,
                          val_years=None, val_size=0.1, shuffle=True,
                          random_seed=None):
    """
    Split targets for train/val/test based on target released year. All
    targets released during <train_years>/<val_years>/<test_years> are included
    in the train/val/test sets respectively. <test_years> cannot be None;
    otherwise, it will throw an assertion.

    If either <train_years> or <val_years> is None, used the remaining targets
    prior to <test_years> for the other set.

    If both <train_years> and <val_years> are None, all targets prior to the min
    of <test_years> are split randomly as train and val sets, using <val_size>
    as the ratio. <val_size> is a float between 0.0 and 1.0 and represent the
    proportion of the train/val targets to include in the val split.
    """
    # Use targets released prior to <test_year_start> for training/validation,
    # and the rest for testing
    assert test_years is not None
    targets_test = targets_df[targets_df.year.isin(test_years)].target.values

    if train_years is not None and val_years is not None:
        targets_train = targets_df[targets_df.year.isin(train_years)].target.values
        targets_val = targets_df[targets_df.year.isin(val_years)].target.values
        return targets_train, targets_val, targets_test

    test_year_start = min(test_years)
    targets_train_val = targets_df[
        targets_df.year < test_year_start].reset_index(drop=True)

    if train_years is None and val_years is None:
        _, val_indices, train_indices = sp.random_split(
            len(targets_train_val), train_split=None, vali_split=val_size,
            test_split=0, shuffle=shuffle, random_seed=random_seed)
        targets_train = targets_train_val.target.values[train_indices]
        targets_val = targets_train_val.target.values[val_indices]

    elif train_years is not None:
        targets_train = targets_train_val[
            targets_train_val.year.isin(train_years)].target.values
        targets_val = targets_train_val[
            ~targets_train_val.year.isin(train_years)].target.values

    elif val_years is not None:
        targets_val = targets_train_val[
            targets_train_val.year.isin(val_years)].target.values
        targets_train = targets_train_val[
            ~targets_train_val.year.isin(val_years)].target.values

    return targets_train, targets_val, targets_test


def generate_train_val_targets_tests(structures_df, targets_train, targets_val,
                                     targets_test, train_decoy_size=None,
                                     val_decoy_size=None, test_decoy_size=None,
                                     exclude_natives=False, random_seed=None):
    """
    Generate train/val/train decoy sets (i.e. a list of structure name
    <target>/<decoy>.pdb). If decoy size is None, include all decoy structures
    for each target; otherwise, sample only that amount for each target. If
    exclude_natives is set, exclude all native structures from the generated set.
    """
    np.random.seed(random_seed)
    sets = [] # [train, val, test]
    if exclude_natives:
        print('Exclude native structures')
        structures_df = structures_df[structures_df.target != structures_df.decoy]
    for targets, decoy_size in zip([targets_train, targets_val, targets_test],
                                   [train_decoy_size, val_decoy_size, test_decoy_size]):
        df = structures_df[structures_df.target.isin(targets)].reset_index(drop=True)
        if decoy_size == None:
            set_df = df
        else:
            gps = df.groupby(['target'])
            idx = np.hstack([np.random.choice(v, decoy_size, replace=False) \
                            for v in gps.groups.values()])
            set_df = df.iloc[idx].reset_index(drop=True)
        sets.append(set_df)
    return sets


def create_sharded_dataset(files, sharded):
    num_shards = sh.get_num_shards(sharded)

    ensembles = en.ensemblers['casp'](files)

    # Check if already partly written.  If so, resume from there.
    metadata_path = sh._get_metadata(sharded)
    if os.path.exists(metadata_path):
        metadata = pd.read_hdf(metadata_path, f'metadata')
        num_written = len(metadata['shard_num'].unique())
    else:
        num_written = 0

    shard_ranges = sh._get_shard_ranges(len(ensembles), num_shards)
    shard_size = shard_ranges[0, 1] - shard_ranges[0, 0]

    total = 0
    print(f'Ensembles per shard: {shard_size:}')
    for shard_num in tqdm.trange(num_written, num_shards):
        start, stop = shard_ranges[shard_num]

        dfs = []
        for name in sorted(ensembles.keys())[start:stop]:
            df = en.parse_ensemble(name, ensembles[name])
            dfs.append(df)
        df = dt.merge_dfs(dfs)

        sh._write_shard(sharded, shard_num, df)


def create_parser():
    parser = argparse.ArgumentParser(description='Generate train/val/test sets.')

    parser.add_argument(
        'target_list',
        help='Path to the file that contains the targets and the years the '
        'targets were released)')
    #parser.add_argument('sharded')
    parser.add_argument('input_dir')
    parser.add_argument('output_sharded_train')
    parser.add_argument('output_sharded_val')
    parser.add_argument('output_sharded_test')
    parser.add_argument('--splitby', '-s', choices=('random', 'year'),
                        default='year', const='year', nargs='?')

    parser.add_argument(
        '--test_years', '-testy', type=int,  nargs='*', default=[2014],
        help='Use all targets released in these years for test set. '
        'Default [2014] (CASP 11)')
    parser.add_argument(
        '--train_years', '-trainy', type=int, nargs='*', default=None,
        help='If not None, use all targets released in these years for train set')
    parser.add_argument(
        '--val_years', '-valy', type=int, nargs='*', default=None,
        help='If not None, Use all targets released in these years for val set')

    parser.add_argument(
        '--train_size', '-train', type=float, default=None,
        help='Fraction of targets to include in train set. If None, '
        'use all remaining. If splitby year, test_size is ignored '
        '(i.e. train_size and val_size should add up 1.0)')
    parser.add_argument(
        '--val_size', '-val', type=float, default=0.1,
        help='Fraction of targets to include in val set')
    parser.add_argument(
        '--test_size', '-test', type=float, default=0.1,
        help='Fraction of targets to include in test set')

    parser.add_argument(
        '--train_decoy_size', '-traind', type=int, default=None,
        help='Number of decoys per target to be included in the train '
        'set. If None, include all decoys')
    parser.add_argument(
        '--val_decoy_size', '-vald', type=int, default=None,
        help='Number of decoys per target to be included in the val '
        'set. If None, include all decoys')
    parser.add_argument(
        '--test_decoy_size', '-testd', type=int, default=None,
        help='Number of decoys per target to be included in the test set. '
        'If None, include all decoys')
    parser.add_argument(
        '--exclude_natives', '-no_nat', action='store_true', default=False,
        help='If set, exclude native structures from the train/val/test set')

    parser.add_argument(
        '--no_shuffle', '-nshuffle', action='store_true', default=False)
    parser.add_argument('--random_seed', '-seed', type=int, default=None)

    return parser


def gen_splits(target_list, input_dir, output_sharded_train, output_sharded_val,
               output_sharded_test, splitby, test_years, train_years, val_years,
               train_size, val_size, test_size,
               train_decoy_size, val_decoy_size, test_decoy_size,
               exclude_natives, shuffle, random_seed):
    """ Generate train/val/test sets from the input dataset. """
    targets_df = pd.read_csv(
        target_list, delimiter='\s*', engine='python').dropna()

    '''structures_df = pd.DataFrame(
        sh.get_names(sharded).apply(lambda x: [util.get_target_name(x),
                                               util.get_decoy_name(x)]).tolist(),
        columns = ['target', 'decoy'])
    structures_df = pd.merge(structures_df, targets_df, on='target')'''


    files = fi.find_files(input_dir, dt.patterns['pdb'])
    structures_df = pd.DataFrame(
        [[util.get_target_name(f), util.get_decoy_name(f), f] for f in files],
        columns = ['target', 'decoy', 'path'])
    # Remove duplicates
    structures_df = structures_df.drop_duplicates(
        subset=['target', 'decoy'], keep='first').reset_index(drop=True)
    structures_df = pd.merge(structures_df, targets_df, on='target')

    # Keep only (target, year) that also appear in structure_df
    targets_df = structures_df[['target', 'year']].drop_duplicates(
        keep='first').reset_index(drop=True)

    if splitby == 'random':
        targets_train, targets_val, targets_test = split_targets_random(
            targets_df, train_size, val_size, test_size, shuffle, random_seed)
    elif splitby == 'year':
        targets_train, targets_val, targets_test = split_targets_by_year(
            targets_df, test_years, train_years, val_years, val_size,
            shuffle, random_seed)
    else:
        assert 'Unrecognized splitby option %s' % splitby

    print('Generating dataset: train ({:} targets), val ({:} targets), '
          'test ({:} targets)'.format(len(targets_train), len(targets_val),
                                      len(targets_test)))

    train_set, val_set, test_set = generate_train_val_targets_tests(
        structures_df, targets_train, targets_val, targets_test,
        train_decoy_size, val_decoy_size, test_decoy_size,
        exclude_natives, random_seed)

    print('Finished generating dataset: train ({:} decoys), val ({:} decoys), '
          'test ({:} decoys)'.format(len(train_set), len(val_set), len(test_set)))

    for (output_sharded, dataset) in [(output_sharded_train, train_set),
                                      (output_sharded_val, val_set),
                                      (output_sharded_test, test_set)]:
        print('\nWriting out dataset to {:}'.format(output_sharded))
        files = dataset.path.unique()
        create_sharded_dataset(files, output_sharded)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    print("\n" + str(args) + "\n")

    gen_splits(
        args.target_list, args.input_dir, args.output_sharded_train,
        args.output_sharded_val, args.output_sharded_test, args.splitby,
        args.test_years, args.train_years, args.val_years,
        args.train_size, args.val_size, args.test_size,
        args.train_decoy_size, args.val_decoy_size, args.test_decoy_size,
        args.exclude_natives, not args.no_shuffle, args.random_seed)
