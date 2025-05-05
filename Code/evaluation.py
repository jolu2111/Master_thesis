from utils import z_score_normalize
import torch
import numpy as np
from exact_sol import damped_harmonic_oscillator

def make_input_params(t_test, values, norm_info=None):
    ''' 
    Makes test parameters for input to the PINN model.
    
    Args:
        t_test: a numpy array of time points to evaluate the model at.
        values: a list of physical values in the order [m, mu, k, y0, v0].
        norm_info: a dictionary with normalization information.
        requires_grad: bool, if True, the resulting tensors will require gradients.
    
    Returns:
        A tuple of torch tensors (m, mu, k, y0, v0) with shape (len(t_test), 1).
    '''
    values = values.copy()
    
    if norm_info is not None:
        values[0] = z_score_normalize(values[0], norm_info['m']) if 'm' in norm_info else values[0]
        values[1] = z_score_normalize(values[1], norm_info['mu']) if 'mu' in norm_info else values[1]
        values[2] = z_score_normalize(values[2], norm_info['k']) if 'k' in norm_info else values[2]
        values[3] = z_score_normalize(values[3], norm_info['y0']) if 'y0' in norm_info else values[3]
        values[4] = z_score_normalize(values[4], norm_info['v0']) if 'v0' in norm_info else values[4]

    m = torch.tensor([values[0]] * len(t_test), dtype=torch.float32).view(-1, 1)
    mu = torch.tensor([values[1]] * len(t_test), dtype=torch.float32).view(-1, 1)
    k = torch.tensor([values[2]] * len(t_test), dtype=torch.float32).view(-1, 1)
    y0 = torch.tensor([values[3]] * len(t_test), dtype=torch.float32).view(-1, 1)
    v0 = torch.tensor([values[4]] * len(t_test), dtype=torch.float32).view(-1, 1)

    return m, mu, k, y0, v0

def limit_state_function_G(model, t, pred_params, differentiable=False, gamma=1000.0,):
    """
    t is a torch array of collocation points.

    Returns:
    g(x) = soft_min(predicted_y) + 1.0  (as a torch tensor)
    """
    # Evaluate the network
    y_pred = model(t, *pred_params)
    
    if differentiable:
        # Compute a smooth approximation to the minimum of y_pred. Differentiable w.r.t. y_pred.
        # soft_min = - (1/gamma)*log(sum(exp(-gamma*y)))
        soft_min = -1.0/gamma * torch.logsumexp(-gamma * y_pred,dim=0)
        
        # The limit state function is then defined as:
        g = soft_min + 1.0
        return g
    else:
        # Compute the minimum of y_pred. Not differentiable w.r.t. y_pred.
        min_y = torch.min(y_pred, dim=0)[0]
        
        # The limit state function is then defined as:
        g = min_y + 1.0
        return g
    
import itertools
def combo_evaluation(PINN_model, specs, norm_info, Num_times=100):
    """
    Evaluate the PINN model for a combination of distributed parameters.
    """

    t_coll = torch.linspace(0, specs['t']['range'], Num_times).view(-1, 1) / norm_info['t']['range']            
    t_test = np.linspace(0, specs['t']['range'], Num_times)

    combo_values = {}
    mean_values = {}
    for param in ['m', 'mu', 'k', 'y0', 'v0']:
        spec = specs[param]
        mean = spec['mean']
        std = spec['std']
        lower = mean + spec['lower_multiplier'] * std
        upper = mean + spec['upper_multiplier'] * std
        
        if param in norm_info:
            combo_value = np.linspace(lower, upper,6) 
            combo_values[param] = combo_value
        else:
            # Use a constant value for non-distributed parameters.
            mean_value = np.array([[mean]])
            mean_values[param] = mean_value

    errors = []
    combos = list(itertools.product(*combo_values.values()))
    print(f"Number of combinations: {len(combos)}")
    for combo in combos:
        input_values =[]
        i=0
        for param in ['m', 'mu', 'k', 'y0', 'v0']:
            if param in norm_info:
                input_values.append(combo[i])
                i+=1
            else:
                input_values.append(mean_values[param].item())  
        test_params = make_input_params(t_coll,input_values, norm_info)
        # Get the PINN prediction
        y_pred = PINN_model(t_coll, *test_params)
        # Compute the exact solution
        y_exact = damped_harmonic_oscillator(t_test, *input_values)
        # Compute the mean squared error for this combination
        mse = np.mean((y_pred.detach().numpy().flatten() - y_exact)**2)
        errors.append(mse)

    combined_mse = np.mean(errors)
    return combined_mse