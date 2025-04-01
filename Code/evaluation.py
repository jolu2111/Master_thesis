from utils import z_score_normalize
import torch

def make_input_params(t_test, values, norm_info=None, requires_grad=False):
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

    m = torch.tensor([values[0]] * len(t_test), dtype=torch.float32).view(-1, 1).requires_grad_(requires_grad)
    mu = torch.tensor([values[1]] * len(t_test), dtype=torch.float32).view(-1, 1).requires_grad_(requires_grad)
    k = torch.tensor([values[2]] * len(t_test), dtype=torch.float32).view(-1, 1).requires_grad_(requires_grad)
    y0 = torch.tensor([values[3]] * len(t_test), dtype=torch.float32).view(-1, 1).requires_grad_(requires_grad)
    v0 = torch.tensor([values[4]] * len(t_test), dtype=torch.float32).view(-1, 1).requires_grad_(requires_grad)

    return m, mu, k, y0, v0
