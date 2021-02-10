import torch
import torch.nn as nn

from cormorant.models import CormorantAtomLevel, CormorantEdgeLevel

from cormorant.nn import MaskLevel, DotMatrix
from cormorant.nn import CatMixReps
from cormorant.cg_lib import CGProduct, CGModule

import logging


class ENN(CGModule):
    """Base equivariant neural network architecture for molecular data.
       Adapted from the class CormorantCG in the Cormorant suite (Anderson et al., 2020). 

    :param maxl: Maximum weight in the output of CG products. (Expanded to list of length :obj:`num_cg_levels`)
    :type maxl: :obj:`int` of :obj:`list` of :obj:`int`
    :param max_sh: Maximum weight in the output of the spherical harmonics. (Expanded to list of length :obj:`num_cg_levels`)
    :type max_sh: :obj:`int` of :obj:`list` of :obj:`int`
    :param num_cg_levels: Number of cg levels to use.
    :type num_cg_levels: :obj:`int`
    :param num_channels: Number of channels that the output of each CG are mixed to. (Expanded to list of length :obj:`num_cg_levels`)
    :type num_channels: :obj:`int` of :obj:`list` of :obj:`int`
    :param level_gain: Gain at each level
    :type level_gain: float 
    :param weight_init: Weight initialization function to use.
    :type weight_init: str
    :param cutoff_type: Types of cutoffs to include.
    :type cutoff_type: str
    :param hard_cut_rad: Radius of HARD cutoff in Angstroms.
    :type hard_cut_rad: float
    :param soft_cut_rad: Radius of SOFT cutoff in Angstroms.
    :type soft_cut_rad: float
    :param soft_cut_width: Width of SOFT cutoff in Angstroms.
    :type soft_cut_width: float
    :param gaussian_mask: Mask using gaussians instead of sigmoids.
    :type gaussian_mask: bool
    :param cgprod_bounded: Put a boundary on the Clebsch-Gordan product 
    :type cgprod_bounded: bool
    :param cg_agg_normalization: normalization for the CG product in the aggregation 
    :type cg_agg_normalization: str
    :param cg_pow_normalization: normalization for the CG product in the
    :type cg_pow_normalization: str
    :param device: Device to initialize the level to.
    :type device: :obj:`torch.device`
    :param dtype: Data type to initialize the level to.
    :type dtype: :obj:`torch.dtype`
    :param cg_dict: Clebsch-Gordan dictionary.
    :type cg_dict: :obj:`nn.cg_lib.CGDict`

    """
    def __init__(self, maxl, max_sh, tau_in_atom, tau_in_edge, tau_pos,
                 num_cg_levels, num_channels, level_gain, weight_init,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False, cgprod_bounded=False,
                 cg_agg_normalization='none', cg_pow_normalization='none',
                 device=None, dtype=None, cg_dict=None):
        # Initialize CG module 
        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict
        # Initialize levels
        atom_levels = nn.ModuleList()
        edge_levels = nn.ModuleList()
        # Initialize taus
        tau_atom, tau_edge = tau_in_atom, tau_in_edge
        # Add levels
        for level in range(num_cg_levels):
            # Add the edge level. Its output type determines the next level.
            edge_lvl = CormorantEdgeLevel(tau_atom, tau_edge, tau_pos[level], num_channels[level], max_sh[level],
                                          cutoff_type, hard_cut_rad[level], soft_cut_rad[level], soft_cut_width[level],
                                          weight_init, gaussian_mask=gaussian_mask, device=device, dtype=dtype)
            edge_levels.append(edge_lvl)
            tau_edge = edge_lvl.tau
            # Add the N-body level.
            atom_lvl = CormorantAtomLevel(tau_atom, tau_edge, maxl[level], num_channels[level+1],
                                          level_gain[level], weight_init, 
                                          cgprod_bounded=cgprod_bounded,
                                          cg_agg_normalization=cg_agg_normalization, 
                                          cg_pow_normalization=cg_pow_normalization,
                                          device=device, dtype=dtype, cg_dict=cg_dict)
            atom_levels.append(atom_lvl)
            tau_atom = atom_lvl.tau
            # Write out taus.
            logging.info('{} {}'.format(tau_atom, tau_edge))
        # Attach levels and taus to the ENN.
        self.max_sh = max_sh
        self.atom_levels = atom_levels
        self.edge_levels = edge_levels
        self.tau_levels_atom = [level.tau for level in atom_levels]
        self.tau_levels_edge = [level.tau for level in edge_levels]

    def forward(self, atom_reps, atom_mask, edge_net, edge_mask, rad_funcs, norms, sph_harm):
        """Run a forward pass of the Cormorant CG layers.

        :param atom_reps: Input atom representations.
        :type atom_reps: SO3 Vector
        :param atom_mask: Batch mask for atom representations. Shape is 
            :math:`(N_{batch}, N_{atom})`. 
        :type atom_mask: :obj:`torch.Tensor` with data type `torch.byte`.
        :param edge_net: Input edge scalar features.
        :type edge_net: SO3 Scalar or None`
        :param edge_mask: Batch mask for atom representations. Shape is 
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        :type edge_mask: :obj:`torch.Tensor` with data type `torch.byte`
        :param rad_funcs: The (possibly learnable) radial filters.
        :type rad_funcs: :obj:`list` of SO3 Scalars
        :param edge_mask: Matrix of the magnitudes of relative position vectors of pairs of atoms.
            :math:`(N_{batch}, N_{atom}, N_{atom})`.    
        :type edge_mask: :obj:`torch.Tensor`
        :param sph_harm: Representation of spherical harmonics calculated from the relative
            position vectors of pairs of points.
        :type sph_harm: SO3 Vector

        :return atoms_all: The concatenated output of the representations output at each level.
        :rtype atoms_all: [SO3 Vector] 
        :return edges_all: The concatenated output of the scalar edge network output at each level.
        :rtype edges_all: [SO3 Scalar]

        """
        assert len(self.atom_levels) == len(self.edge_levels) == len(rad_funcs)
        # Iinitialize representations
        atoms_all = []
        edges_all = []
        # Construct iterated multipoles
        for idx, (atom_level, edge_level, max_sh) in enumerate(zip(self.atom_levels, self.edge_levels, self.max_sh)):
            # Edge representations
            edge_net = edge_level(edge_net, atom_reps, rad_funcs[idx], edge_mask, norms)
            edge_reps = edge_net * sph_harm[:max_sh+1]
            # Atom representations
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)
            # Append the representations
            atoms_all.append(atom_reps)
            edges_all.append(edge_net)
        return atoms_all, edges_all

