import torch
import torch.nn as nn

import logging

from cormorant.cg_lib import CGModule, SphericalHarmonicsRel

from cormorant.models.cormorant_cg import CormorantCG

from cormorant.nn import RadialFilters
from cormorant.nn import InputLinear, InputMPNN
from cormorant.nn import OutputLinear, OutputPMLP, OutputSoftmax, GetScalarsAtom
from cormorant.nn import NoLayer


class CormorantMutation(CGModule):
    """
    Basic Cormorant Network used to train on BDBBind.

    Parameters
    ----------
    maxl : :obj:`int` of :obj:`list` of :obj:`int`
        Maximum weight in the output of CG products. (Expanded to list of
        length :obj:`num_cg_levels`)
    max_sh : :obj:`int` of :obj:`list` of :obj:`int`
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length :obj:`num_cg_levels`)
    num_cg_levels : :obj:`int`
        Number of cg levels to use.
    num_channels : :obj:`int` of :obj:`list` of :obj:`int`
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length :obj:`num_cg_levels`)
    num_species : :obj:`int`
        Number of species of atoms included in the input dataset.

    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to level to
    cg_dict : :obj:`nn.cg_lib.CGDict`
    """
    def __init__(self, maxl, max_sh, num_cg_levels, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask, num_classes=2, 
                 device=None, dtype=None, cg_dict=None):

        logging.info('Initializing network!')
        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels+1)

        logging.info('hard_cut_rad: {}'.format(hard_cut_rad))
        logging.info('soft_cut_rad: {}'.format(soft_cut_rad))
        logging.info('soft_cut_width: {}'.format(soft_cut_width))
        logging.info('maxl: {}'.format(maxl))
        logging.info('max_sh: {}'.format(max_sh))
        logging.info('num_channels: {}'.format(num_channels))

        super().__init__(maxl=max(maxl+max_sh), device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.num_species = num_species

        # Set up spherical harmonics
        self.sph_harms = SphericalHarmonicsRel(max(max_sh), conj=True,
                                               device=device, dtype=dtype, cg_dict=cg_dict)

        # Set up position functions, now independent of spherical harmonics
        self.rad_funcs = RadialFilters(max_sh, basis_set, num_channels, num_cg_levels,
                                       device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]

        self.input_func_atom = InputLinear(num_scalars_in, num_scalars_out,
                                           device=self.device, dtype=self.dtype)
        self.input_func_edge = NoLayer()

        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau

        self.cormorant_cg = CormorantCG(maxl, max_sh, tau_in_atom, tau_in_edge,
                     tau_pos, num_cg_levels, num_channels, level_gain, weight_init,
                     cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                     cat=True, gaussian_mask=False,
                     device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_atom = self.cormorant_cg.tau_levels_atom
        tau_cg_levels_edge = self.cormorant_cg.tau_levels_edge

        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom,
                                               device=self.device, dtype=self.dtype)
        self.get_scalars_edge = NoLayer()

        num_scalars_atom = self.get_scalars_atom.num_scalars
        num_scalars_edge = self.get_scalars_edge.num_scalars

        self.output_layer_atom = OutputSoftmax(num_scalars_atom, num_classes, bias=True,
                                               device=self.device, dtype=self.dtype) 
        self.output_layer_edge = NoLayer()

        logging.info('Model initialized. Number of parameters: {}'.format(
            sum([p.nelement() for p in self.parameters()])))


    def forward_once(self, data):
        """
        Runs a single forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer
        """
        # Get and prepare the data
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        spherical_harmonics, norms = self.sph_harms(atom_positions, atom_positions)
        rad_func_levels = self.rad_funcs(norms, edge_mask * (norms > 0))

        # Prepare the input reps for both the atom and edge network
        atom_reps_in = self.input_func_atom(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # Clebsch-Gordan layers central to the network
        atoms_all, edges_all = self.cormorant_cg(atom_reps_in, atom_mask, edge_net_in, edge_mask,
                                                 rad_func_levels, norms, spherical_harmonics)

        # Construct scalars for network output
        atom_scalars = self.get_scalars_atom(atoms_all)
        edge_scalars = self.get_scalars_edge(edges_all)

        # Prediction in this case will depend only on the atom_scalars. Can make
        # it more general here.
        prediction = self.output_layer_atom(atom_scalars, atom_mask)

        return prediction, atoms_all, edges_all
 

    def forward(self, data, covariance_test=False):
        """
        Runs a single forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction1 : :obj:`torch.Tensor`
            The output of the first network
        prediction2 : :obj:`torch.Tensor`
            The output of the second network
        """

        print('Data:', data.keys())

        data1 = {}
        data2 = {}
        data1['label'] = data['label']
        data2['label'] = data['label']
        data1['charges']   = data['charges1']
        data2['charges']   = data['charges2']
        data1['positions'] = data['positions1']
        data2['positions'] = data['positions2']
        data1['one_hot']   = data['one_hot1']
        data2['one_hot']   = data['one_hot2']
        data1['atom_mask'] = data['atom_mask1']
        data2['atom_mask'] = data['atom_mask2']
        data1['edge_mask'] = data['edge_mask1']
        data2['edge_mask'] = data['edge_mask2']

        print('charges 1:',   data1['charges'].shape)
        print('charges 2:',   data2['charges'].shape)
        print('positions 1:', data1['positions'].shape)
        print('positions 2:', data2['positions'].shape)
        print('one_hot 1:',   data1['one_hot'].shape)
        print('one_hot 2:',   data2['one_hot'].shape)
        print('atom_mask 1:', data1['atom_mask'].shape)
        print('atom_mask 2:', data2['atom_mask'].shape)
        print('edge_mask 1:', data1['edge_mask'].shape)
        print('edge_mask 2:', data2['edge_mask'].shape)

        prediction1, atoms_all1, edges_all1 = self.forward_once(data1)
        prediction2, atoms_all2, edges_all2 = self.forward_once(data2)

        print('prediction1:', prediction1)
        print('prediction2:', prediction2)

        prediction = (prediction2 - prediction1)**2

        # Covariance test
        if covariance_test:
            return prediction, atoms_all1, edges_all1
        else:
            return prediction


    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_positions: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        charge_power, charge_scale, device, dtype = self.charge_power, self.charge_scale, self.device, self.dtype

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        atom_mask = data['atom_mask'].to(device)
        edge_mask = data['edge_mask'].to(device)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions


def expand_var_list(var, num_cg_levels):
    if type(var) is list:
        var_list = var + (num_cg_levels-len(var))*[var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list


