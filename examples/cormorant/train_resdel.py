#
# Cormorant PDBBind training script
#

import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from cormorant_pdbbind import CormorantPDBBind
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():

    # Initialize arguments -- Just
    args = init_argparse('resdel')

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloader
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'resdel', 
                                                                    force_download=args.force_download,
                                                                    ignore_check=args.ignore_check
                                                                    )

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}

    # Initialize model
    model = CormorantResDel(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                             args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                             args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                             charge_scale, args.gaussian_mask,
                             args.top, args.input, args.num_mpnn_levels,
                             device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define cross-entropy as the loss function.
    loss_fn = torch.nn.functional.cross_entropy

    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype, task='classification', clip_value=0.1)
    print('Initialized a',task,'trainer with clip value',trainer.clip_value)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()

if __name__ == '__main__':
    main()

