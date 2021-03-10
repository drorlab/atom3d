import torch
import torch.nn as nn

import logging

from cormorant.cg_lib import CGModule, SphericalHarmonicsRel

from cormorant.nn import RadialFilters
from cormorant.nn import InputLinear, InputMPNN
from cormorant.nn import OutputLinear, OutputPMLP, OutputSoftmax, OutputSoftmaxPMLP
from cormorant.nn import GetScalarsAtom
from cormorant.nn import NoLayer

from atom3d.models.enn import ENN


class ENN_MSP(CGModule):
    """
    Cormorant Network used to train on MSP.

    :param maxl: Maximum weight in the output of CG products. (Expanded to list of length :obj:`num_cg_levels`)
    :type maxl: :obj:`int` of :obj:`list` of :obj:`int`
    :param max_sh: Maximum weight in the output of the spherical harmonics  (Expanded to list of length :obj:`num_cg_levels`)
    :type max_sh: :obj:`int` of :obj:`list` of :obj:`int`
    :param num_cg_levels: Number of cg levels to use.
    :type num_cg_levels: :obj:`int`
    :param num_channels: Number of channels that the output of each CG are mixed to (Expanded to list of length :obj:`num_cg_levels`)
    :type num_channels: :obj:`int` of :obj:`list` of :obj:`int`
    :param num_species: Number of species of atoms included in the input dataset.
    :type num_species: :obj:`int`
    :param device: Device to initialize the level to.
    :type device: :obj:`torch.device`
    :param dtype: Data type to initialize the level to level to.
    :type dtype: :obj:`torch.dtype`
    :param cg_dict: Clebsch-Gordan dictionary.
    :type cg_dict: :obj:`nn.cg_lib.CGDict`

    """
    def __init__(self, maxl, max_sh, num_cg_levels, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask, 
                 num_mpnn_layers=64, activation='leakyrelu', num_classes=2, 
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

        # Set up input layers
        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]
        self.input_func_atom = InputLinear(num_scalars_in, num_scalars_out,
                                           device=self.device, dtype=self.dtype)
        self.input_func_edge = NoLayer()

        # Set up the central Clebsch-Gordan network
        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau
        self.cormorant_cg = ENN(maxl, max_sh, tau_in_atom, tau_in_edge, tau_pos, 
                                num_cg_levels, num_channels, level_gain, weight_init,
                                cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                                cat=True, gaussian_mask=False, cgprod_bounded=True,
                                cg_agg_normalization='none', cg_pow_normalization='none',
                                device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        # Get atom and edge scalars
        tau_cg_levels_atom = self.cormorant_cg.tau_levels_atom
        tau_cg_levels_edge = self.cormorant_cg.tau_levels_edge
        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom, device=self.device, dtype=self.dtype)
        self.get_scalars_edge = NoLayer()

        # Set up the output networks
        num_scalars_atom = self.get_scalars_atom.num_scalars
        num_scalars_edge = self.get_scalars_edge.num_scalars
        self.output_layer_atom = OutputSoftmaxPMLP(num_scalars_atom, num_classes,
                                                   num_mixed=num_mpnn_layers, activation=activation,
                                                   device=self.device, dtype=self.dtype) 
        self.output_layer_edge = NoLayer()

        logging.info('Model initialized. Number of parameters: {}'.format(
            sum([p.nelement() for p in self.parameters()])))


    def forward_once(self, data):
        """
        Runs a single forward pass of the network.

        :param data: Dictionary of data to pass to the network.
        :type data : :obj:`dict`
        :param covariance_test: If true, returns all of the atom-level representations twice.
        :type covariance_test: :obj:`bool`, optional
            
        :return prediction: The output of the layer
        :rtype prediction: :obj:`torch.Tensor`
            
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
        return atom_scalars, edge_scalars, atoms_all, edges_all, atom_mask, edge_mask
        # Prediction in this case will depend only on the atom_scalars. 
        # Can make it more general here.
        #prediction = self.output_layer_atom(atom_scalars, atom_mask)
        #prediction, atoms_all, edges_all
 

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        :param data: Dictionary of data to pass to the network.
        :type data : :obj:`dict`
        :param covariance_test: If true, returns all of the atom-level representations twice.
        :type covariance_test: :obj:`bool`, optional
            
        :return prediction: The output of the first network.
        :rtype prediction: :obj:`torch.Tensor`
            
        """
        # Split the data
        data1 = {'label': data['label']}
        data2 = {'label': data['label']}
        for key in ['charges', 'positions', 'one_hot', 'atom_mask', 'edge_mask']:
            data1[key] = data[key+'1']
            data2[key] = data[key+'2']
        # Run the two separate networks
        atom_scalars1, edge_scalars2, atoms_all1, edges_all1, atom_mask1, edge_mask1 = self.forward_once(data1)
        atom_scalars2, edge_scalars2, atoms_all2, edges_all2, atom_mask2, edge_mask2 = self.forward_once(data2)
        # Combine atom scalars
        atom_scalars = torch.cat((atom_scalars1, atom_scalars2), dim=1)
        atom_mask = torch.cat((atom_mask1, atom_mask2), dim=1)
        # Apply the output layer
        prediction = self.output_layer_atom(atom_scalars, atom_mask)
        # Covariance test
        if covariance_test:
            return prediction, atoms_all1, edges_all1
        else:
            return prediction


    def prepare_input(self, data):
        """
        Extracts input from data class.

        :param data: Information on the state of the system.
        :type data: dict

        :return atom_scalars: Tensor of scalars for each atom.
        :rtype atom_scalars: :obj:`torch.Tensor`
        :return atom_mask: Mask used for batching data.
        :rtype atom_mask: :obj:`torch.Tensor`
        :return atom_positions: Positions of the atoms.
        :rtype atom_positions: :obj:`torch.Tensor`
        :return edge_mask: Mask used for batching data.
        :rtype edge_mask: :obj:`torch.Tensor`
            
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


