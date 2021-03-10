import logging
import gc
# PyTorch modules
import torch
from torch.utils.data import DataLoader
# Cormorant modules and functions
from cormorant.engine import Engine
from cormorant.engine import init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.models.autotest import cormorant_tests
# Functions that have been adapted from cormorant functions
from utils import init_cormorant_argparse, init_cormorant_file_paths
# SMP-specific model in ATOM3D
from model import ENN_MSP
# Modules to load and handle SMP data
from data import collate_msp, initialize_msp_data

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():
    # Initialize arguments
    args = init_cormorant_argparse('msp')
    # Initialize file paths
    args = init_cormorant_file_paths(args)
    # Initialize logger
    init_logger(args)
    # Initialize device and data type
    device, dtype = init_cuda(args)
    # Initialize dataloader # Use initialize_msp_data to load LMDB directly (needs much more memory)
    init = initialize_msp_data(args, args.datadir)
    args, datasets, num_species, charge_scale = init
    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers, collate_fn=collate_msp)
                   for split, dataset in datasets.items()}
    # Initialize model
    model = ENN_MSP(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                    args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                    args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                    charge_scale, args.gaussian_mask, num_classes=args.num_classes,
                    device=device, dtype=dtype)
    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)
    # Define cross-entropy as the loss function.
    loss_fn = torch.nn.functional.cross_entropy
    gc.collect()
    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale, siamese=True)
    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, 
                     device, dtype, task='classification', clip_value=None)
    print('Initialized a',trainer.task,'trainer.')
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()
    # Train model.
    trainer.train()
    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()


if __name__ == '__main__':
    main()


