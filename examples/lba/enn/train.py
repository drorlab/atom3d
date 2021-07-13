import logging
# PyTorch modules
import torch
from torch.utils.data import DataLoader
# Cormorant modules and functions
from cormorant.data.utils import initialize_datasets
from cormorant.engine import Engine
from cormorant.engine import init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.models.autotest import cormorant_tests
# Functions that have been adapted from cormorant functions
from utils import init_cormorant_argparse, init_cormorant_file_paths
# LBA-specific model 
from model import ENN_LBA, ENN_LBA_Siamese
# Methods to load and handle LBA data
from data import CormorantDatasetLBA, collate_lba, collate_lba_siamese, initialize_lba_data

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():
    # Initialize arguments
    args = init_cormorant_argparse('lba')
    # Initialize file paths
    args = init_cormorant_file_paths(args)
    # Initialize logger
    init_logger(args)
    # Initialize device and data type
    device, dtype = init_cuda(args)
    # Initialize dataloader
    args, datasets, num_species, charge_scale = initialize_lba_data(args, args.datadir)        
    # Further differences for Siamese networks
    if args.siamese:
        collate_fn = collate_lba_siamese
    else:
        collate_fn = collate_lba
    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers, collate_fn=collate_fn)
                   for split, dataset in datasets.items()}
    # Initialize model
    if args.siamese:
        model = ENN_LBA_Siamese(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                        args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                        args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                        charge_scale, args.gaussian_mask, cgprod_bounded=args.cgprod_bounded,
                        cg_pow_normalization=args.cg_pow_normalization, cg_agg_normalization=args.cg_agg_normalization,
                        device=device, dtype=dtype)
    else:
        model = ENN_LBA(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                        args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                        args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                        charge_scale, args.gaussian_mask, cgprod_bounded=args.cgprod_bounded,
                        cg_pow_normalization=args.cg_pow_normalization, cg_agg_normalization=args.cg_agg_normalization,
                        device=device, dtype=dtype)
    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)
    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss
    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale, siamese = args.siamese)
    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype, clip_value=None)
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()
    # Train model.
    trainer.train()
    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()


if __name__ == '__main__':
    main()


