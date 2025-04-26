import torch

def z_score_normalize_simple(x, mean, std):
    """Applies Z-score normalization."""
    return (x - mean) / std

def initialize_param(N, distribute=None, specs=None, normalize=False):
    """
    Initialize training parameters with options to distribute selected parameters.

    Parameters:
    -----------
    N : int
        Number of samples for distributed parameters.
    distribute : list or None
        List of parameter names to be distributed (sampled). For example, ['m', 'mu'].
        Parameters not included here will be set as constants (their mean values).
        If None, no parameters are distributed.
    specs : dict or None
        A dictionary mapping parameter names to a dict with keys:
            'mean': float, the parameter's mean.
            'std': float, the parameter's standard deviation.
            'lower_multiplier': float, multiplier to compute the lower bound (lower = mean + lower_multiplier * std).
            'upper_multiplier': float, multiplier to compute the upper bound (upper = mean + upper_multiplier * std).
        If None, default specifications are used.
    normalize : bool
        Whether to apply Z-score normalization to the sampled (distributed) parameters.

    Example:
    --------
    To distribute only mass and mu:
    
    >>> specs = {
    ...     'm':  {'mean': 1.0,  'std': 0.1, 'lower_multiplier': -2, 'upper_multiplier': 8},
    ...     'mu': {'mean': 0.6,  'std': 0.1, 'lower_multiplier': -8, 'upper_multiplier': 2},
    ...     'k':  {'mean': 5.0,  'std': 0.5, 'lower_multiplier': -5, 'upper_multiplier': 5},
    ...     'y0': {'mean': -0.4, 'std': 0.1, 'lower_multiplier': -5, 'upper_multiplier': 5},
    ...     'v0': {'mean': 3.0,  'std': 0.5, 'lower_multiplier': -5, 'upper_multiplier': 5},
    ... }
    >>> params = initialize_param(N=100, distribute=['m', 'mu'], specs=specs, normalize=True)
    
    """
    ################# Define default specifications for parameters if specs is not provided. #######################
    default_specs = {
        't': {'range': 5.0},
        'm':  {'mean': 1.0,  'std': 0.1, 'lower_multiplier': -5, 'upper_multiplier': 5},
        'mu': {'mean': 0.6,  'std': 0.05, 'lower_multiplier': -5, 'upper_multiplier': 5},
        'k':  {'mean': 5.0,  'std': 0.5, 'lower_multiplier': -5, 'upper_multiplier': 5},
        'y0': {'mean': -0.4, 'std': 0.1, 'lower_multiplier': -5, 'upper_multiplier': 5},
        'v0': {'mean': 3.0,  'std': 0.5, 'lower_multiplier': -5, 'upper_multiplier': 5},
    }

    if specs is None:
        specs = default_specs
    else:
    # Fill in any missing parameter specs with the defaults.
        for key, val in default_specs.items():
            if key not in specs:
                specs[key] = val
    
    ################## Create time collocation points ############################################################
    if normalize: # If normalization is used, the time collocation points should be normalized as well.
        t_coll = torch.rand(N, 1)
        t_coll.requires_grad_(True)
    else:
        t_coll = torch.rand(N, 1)*specs['t']['range']
        t_coll.requires_grad_(True)

    # t_coll = torch.rand(N, 1)*specs['t']['range']
    # t_coll.requires_grad_(True)


    # Initialize the output dictionary with the time collocation points.
    params = {'t_coll': t_coll}
    
    ############# Initialize parameters based on which to distribution and normalize ##############################

    # If distribute is not provided, assume no parameter is distributed.
    if distribute is None:
        distribute = []

    # Also prepare a normalization info dictionary if normalization is used.

    if normalize:
        norm_info = {}
        norm_info['t'] = specs['t']
        # Add normalization info and flag to the dictionary.
        params['norm_info'] = norm_info
    else:
        params['norm_info'] = None
    
    # For each parameter, either sample uniformly (if distributed) or use a constant value.
    for param in ['m', 'mu', 'k', 'y0', 'v0']:
        spec = specs[param]
        mean = spec['mean']
        std = spec['std']
        lower = mean + spec['lower_multiplier'] * std
        upper = mean + spec['upper_multiplier'] * std
        
        if param in distribute:
            # Sample uniformly for distributed parameters.
            tensor = torch.FloatTensor(N, 1).uniform_(lower, upper) 
            if normalize:
                # Apply Z-score normalization
                tensor = z_score_normalize_simple(tensor, mean, std)
                # Store normalization info for later inversion in the loss functions.
                norm_info[param] = {'mean': mean, 'std': std}
            tensor.requires_grad_(True)
        else:
            # Use a constant value for non-distributed parameters.
            tensor = torch.tensor([[mean]], requires_grad=True)
            tensor = tensor.expand(N, -1)
        params[param] = tensor

    # Define a single time for the boundary condition (same shape as t_coll)
    t0 = torch.zeros_like(t_coll).clone().detach().requires_grad_(True)
    params['t0'] = t0
    
    return params
