import argparse
import os, sys, pickle
from datetime import datetime
from math import inf, log, log2, exp, ceil

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from cormorant.engine.args import setup_shared_args, BoolArg, Range

import logging
logger = logging.getLogger(__name__)



# -- UTILS FOR ARGUMENT PARSING --


def init_cormorant_argparse(dataset):
    """
    Set up the argparse object for a specific dataset.

    :param dataset: Dataset being used.
    :type dataset: str

    :return args: Namespace with a dictionary of arguments where the key is the name of the argument and the item is the input value.
    :rtype args: Namespace
    
    """
    parser = argparse.ArgumentParser(description='Cormorant network options.')
    parser = setup_shared_args(parser)
    parser.add_argument('--format', type=str, default='npz', help='Input data format.')
    if dataset.lower() in ["smp"]:
        pass
    elif dataset == "lba":
        parser.add_argument('--radius', type=float, default=6.,
                            help='radius of the selected region around the ligand.')
        parser.add_argument('--maxnum', type=float, default=500,
                            help='maximum total number of atoms of the ligand + the region around it.')
        parser.add_argument('--siamese', action=BoolArg, default=False,
                            help='use a Siamese architecture.')
    elif dataset == "res":
        parser.add_argument('--num_classes', type=int, default=20,
                            help='number of classes for the classification.')
        parser.add_argument('--maxnum', type=float, default=400,
                            help='maximum total number of atoms of the ligand + the region around it.')
        parser.add_argument('--samples', type=float, default=100,
                            help='maximum number of protein structures to use per split.')
    elif dataset == "lep":
        parser.add_argument('--num_classes', type=int, default=2,
                            help='number of classes for the classification.')
        parser.add_argument('--radius', type=float, default=6.,
                            help='radius of the selected region around the ligand.')
        parser.add_argument('--maxnum', type=float, default=400,
                            help='maximum total number of atoms of the ligand + the region around it.')
        parser.add_argument('--droph', action=BoolArg, default=False,
                            help='drop hydrogen atoms.')
    elif dataset == "msp":
        parser.add_argument('--num_classes', type=int, default=2,
                            help='number of classes for the classification.')
        parser.add_argument('--radius', type=float, default=6.,
                            help='radius of the selected region around the mutated residue.') 
        parser.add_argument('--droph', action=BoolArg, default=False,
                            help='drop hydrogen atoms.')
    else: raise ValueError("Dataset %s is not recognized."%dataset)
    args = parser.parse_args()
    d = vars(args)
    d['dataset'] = dataset
    return args


def init_cormorant_file_paths(args):
    """
    Set up the file paths for Cormorant.
    """
    # Initialize files and directories to load/save logs, models, and predictions
    workdir = args.workdir
    prefix = args.prefix
    modeldir = args.modeldir
    logdir = args.logdir
    predictdir = args.predictdir
    if prefix and not args.logfile: args.logfile = os.path.join(workdir, logdir, prefix+'.log')
    if prefix and not args.bestfile: args.bestfile = os.path.join(workdir, modeldir, prefix+'_best.pt')
    if prefix and not args.checkfile: args.checkfile = os.path.join(workdir, modeldir, prefix+'.pt')
    if prefix and not args.loadfile: args.loadfile = args.checkfile
    if prefix and not args.predictfile: args.predictfile = os.path.join(workdir, predictdir, prefix)
    # Create ouput directories
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
        if not args.target: args.target = 'gap'
    elif args.dataset.lower().startswith('lba'):
        if not args.target: args.target = 'neglog_aff'
    elif args.dataset.lower().startswith('res'):
        if not args.target: args.target = 'label'
    elif args.dataset.lower().startswith('msp'):
        if not args.target: args.target = 'label'
    elif args.dataset.lower().startswith('lep'):
        if not args.target: args.target = 'label'
    else:
        raise ValueError('Dataset %s not recognized!'%args.dataset)
    # Log information
    logger.info('Initializing simulation based upon argument string:')
    logger.info(' '.join([arg for arg in sys.argv]))
    logger.info('Log, best, checkpoint, load files: {} {} {} {}'.format(args.logfile, args.bestfile, args.checkfile, args.loadfile))
    logger.info('Dataset, learning target, datadir: {} {} {}'.format(args.dataset, args.target, args.datadir))
    # Set the random seed
    if args.seed < 0:
        seed = int((datetime.now().timestamp())*100000)
        logger.info('Setting seed based upon time: {}'.format(seed))
        args.seed = seed
        torch.manual_seed(seed)
    return args


# -- UTILS FOR COLLATE FUNCTIONS --


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the largest tensor along each axis.

    :param props: Pytorch tensors to stack
    :type props: list of Pytorch Tensors

    :return props: Stacked pytorch tensor
    :rtype props: Pytorch tensor.

    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, key, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    :param props: Full Dataset
    :type props: Pytorch tensor

    :return props: The dataset with only the retained information.
    :rtype props: Pytorch tensor

    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    elif key == 'bonds':
        return props[:, to_keep, ...][:, :, to_keep, ...]
    else:
        return props[:, to_keep, ...]


