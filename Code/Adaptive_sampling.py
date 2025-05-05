# Generate latin hypercube samples
import numpy as np
import torch
from scipy.stats import qmc

def generate_latin_hypercube_samples(specs, norm_info, num_samples=1000):
    """
    Generate Latin Hypercube samples for parameters in specs that also is in params['norm_info].

    returns:
        - samples: A numpy array with a coloumn for each parameter in specs. 
        The params that are in norm_info are sampled, and the rest are set to their mean value.
        - params_to_sample: A list of the parameters that were sampled.
    """ 
    # Extract the number of samples and the parameters to sample from specs
    params_to_sample = [param for param in norm_info.keys() if param != 't']   #### Changed this without testing if it works #####
    
    # Create a Latin Hypercube sampler
    sampler = qmc.LatinHypercube(len(params_to_sample))
    samples = sampler.random(num_samples)
    
    # Scale the samples to the ranges defined in specs
    scaled_samples = []
    i = 0
    for param in specs.keys():
        if param == 't':
            continue
        if param in params_to_sample:
            lower_bound = specs[param]['mean'] + specs[param]['lower_multiplier'] * specs[param]['std']
            upper_bound = specs[param]['mean'] + specs[param]['upper_multiplier'] * specs[param]['std']
            scaled_samples.append(qmc.scale(samples[:, i].reshape(-1, 1), lower_bound, upper_bound))
            i += 1
        else:
            # If the parameter is not in the specs, just use its mean value
            scaled_samples.append(np.full((num_samples, 1), specs[param]['mean']))
    return np.array(scaled_samples).T[0], params_to_sample

def generate_sobol_samples(specs, norm_info, num_samples=1000):
    """
    Generate Sobol samples for parameters in specs that also is in params['norm_info].

    returns:
        - samples: A numpy array with a coloumn for each parameter in specs. 
        The params that are in norm_info are sampled, and the rest are set to their mean value.
        - params_to_sample: A list of the parameters that were sampled.
    """ 
    # Extract the number of samples and the parameters to sample from specs
    params_to_sample = [param for param in specs.keys() if param in norm_info and param != 't']
    
    # Create a Sobol sampler
    sampler = qmc.Sobol(len(params_to_sample))
    samples = sampler.random(num_samples)
    
    # Scale the samples to the ranges defined in specs
    scaled_samples = []
    i = 0
    for param in specs.keys():
        if param == 't':
            continue
        if param in params_to_sample:
            lower_bound = specs[param]['mean'] + specs[param]['lower_multiplier'] * specs[param]['std']
            upper_bound = specs[param]['mean'] + specs[param]['upper_multiplier'] * specs[param]['std']
            scaled_samples.append(qmc.scale(samples[:, i].reshape(-1, 1), lower_bound, upper_bound))
            i += 1
        else:
            # If the parameter is not in the specs, just use its mean value
            scaled_samples.append(np.full((num_samples, 1), specs[param]['mean']))
    return np.array(scaled_samples).T[0], params_to_sample

def make_new_params(points_selected, specs, params):
    """
    Create new parameters for the PINN model based on selected points.

    Args:
        points_selected: A numpy array of shape (N, 5) with the selected points.
        specs: A dictionary containing the specifications for each parameter.
        params: A dictionary containing the current parameters.

    Returns:
        new_params: A dictionary with the new parameters for the PINN model.
    """
    # Create a copy of the original params to avoid modifying it directly
    new_params = params.copy()

    # Normalize the values in points_selected for the parameters in norm_info
    normalized_values = points_selected.copy()
    for i, param in enumerate(['m', 'mu', 'k', 'y0', 'v0']):
        if param in params['norm_info']:
            mean = specs[param]['mean']
            std = specs[param]['std']
            normalized_values[:, i] = (points_selected[:, i] - mean) / std

    # Substitute the normalized values into new_params
    new_params['m'] = torch.tensor(normalized_values[:, 0], dtype=torch.float32).view(-1, 1)
    new_params['mu'] = torch.tensor(normalized_values[:, 1], dtype=torch.float32).view(-1, 1)
    new_params['k'] = torch.tensor(normalized_values[:, 2], dtype=torch.float32).view(-1, 1)
    new_params['y0'] = torch.tensor(normalized_values[:, 3], dtype=torch.float32).view(-1, 1)
    new_params['v0'] = torch.tensor(normalized_values[:, 4], dtype=torch.float32).view(-1, 1)
    # Create a new t_coll and t0 with the same length as the other values
    num_samples = normalized_values.shape[0]
    new_params['t_coll'] = torch.rand(num_samples, 1, dtype=torch.float32).requires_grad_(True)
    new_params['t0'] = torch.zeros(num_samples, 1, dtype=torch.float32).requires_grad_(True)
    new_params['norm_info'] = params['norm_info']  # Keep the normalization info

    return new_params