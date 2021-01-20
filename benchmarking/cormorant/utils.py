import argparse
import os, sys, pickle
from datetime import datetime
from math import inf, log, log2, exp, ceil

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from cormorant.engine.args import setup_shared_args

import logging
logger = logging.getLogger(__name__)



def init_cormorant_argparse(dataset):
    """
    Sets up the argparse object for a specific dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.

    Returns
    -------
    args : :class:`Namespace`
        Namespace with a dictionary of arguments where the key is the name of
        the argument and the item is the input value.
    """
    parser = argparse.ArgumentParser(description='Cormorant network options.')
    parser = setup_shared_args(parser)
    # Datasets without additional arguments
    if dataset.lower() in ["smp", "lba"]:
        pass
    # Datasets for classification tasks
    elif dataset == "res":
        parser.add_argument('--num_classes', type=int, default=20,
                            help='number of classes for the classification.')
    elif dataset in ["msp", "lep"]:
        parser.add_argument('--num_classes', type=int, default=2,
                            help='number of classes for the classification.')
    else:
        raise ValueError("Dataset %s is not recognized."%dataset)
    args = parser.parse_args()
    d = vars(args)
    d['dataset'] = dataset
    return args


def init_cormorant_file_paths(args):

    # Initialize files and directories to load/save logs, models, and predictions
    workdir = args.workdir
    prefix = args.prefix
    modeldir = args.modeldir
    logdir = args.logdir
    predictdir = args.predictdir

    if prefix and not args.logfile:  args.logfile =  os.path.join(workdir, logdir, prefix+'.log')
    if prefix and not args.bestfile: args.bestfile = os.path.join(workdir, modeldir, prefix+'_best.pt')
    if prefix and not args.checkfile: args.checkfile = os.path.join(workdir, modeldir, prefix+'.pt')
    if prefix and not args.loadfile: args.loadfile = args.checkfile
    if prefix and not args.predictfile: args.predictfile = os.path.join(workdir, predictdir, prefix)

    if not os.path.exists(modeldir):
        logger.warning('Model directory {} does not exist. Creating!'.format(modeldir))
        os.mkdir(modeldir)
    if not os.path.exists(logdir):
        logger.warning('Logging directory {} does not exist. Creating!'.format(logdir))
        os.mkdir(logdir)
    if not os.path.exists(predictdir):
        logger.warning('Prediction directory {} does not exist. Creating!'.format(predictdir))
        os.mkdir(predictdir)

    # Set standard targets
    if args.dataset.lower().startswith('smp'):
        if not args.target:
            args.target = 'gap'
    elif args.dataset.lower().startswith('lba'):
       if not args.target:
            args.target = 'neglog_aff'
    elif args.dataset.lower().startswith('res'):
       if not args.target:
            args.target = 'residue'
    elif args.dataset.lower().startswith('msp'):
       if not args.target:
            args.target = 'label'
    elif args.dataset.lower().startswith('lep'):
       if not args.target:
            args.target = 'label'
    else:
        raise ValueError('Dataset %s not recognized!'%args.dataset)

    logger.info('Initializing simulation based upon argument string:')
    logger.info(' '.join([arg for arg in sys.argv]))
    logger.info('Log, best, checkpoint, load files: {} {} {} {}'.format(args.logfile, args.bestfile, args.checkfile, args.loadfile))
    logger.info('Dataset, learning target, datadir: {} {} {}'.format(args.dataset, args.target, args.datadir))

    if args.seed < 0:
        seed = int((datetime.now().timestamp())*100000)
        logger.info('Setting seed based upon time: {}'.format(seed))
        args.seed = seed
        torch.manual_seed(seed)

    return args

