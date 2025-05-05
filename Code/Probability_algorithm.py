import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import qmc
from batching import make_input_params_batched, limit_state_function_G_batched
from exact_sol import damped_harmonic_oscillator

def generate_normal_samples(specs, norm_info, num_samples=1000):
    """
    Generate samples for parameters in specs that follow the normal distributions 
    defined in norm_info and specs.

    returns:
        - samples: A numpy array with a column for each parameter in specs. 
                  The params that are in norm_info are sampled from normal distributions,
                  and the rest are set to their mean value.
        - params_to_sample: A list of the parameters that were sampled.
    """ 
    # Extract the parameters that need to be sampled based on norm_info
    params_to_sample = [param for param in specs.keys() if param in norm_info and param != 't']
    
    # Initialize a list to hold the sampled values
    sampled_params = []

    for param in specs.keys():
        if param == 't':
            continue
        
        if param in params_to_sample:
            # For sampled parameters, use normal distribution
            mean = specs[param]['mean']
            std = specs[param]['std']
            samples = np.random.normal(mean, std, num_samples)
            
            # # Optionally, clip the samples to ensure they fall within a reasonable range
            # # You can adjust these ranges or skip clipping depending on your problem's needs
            # lower_bound = mean + specs[param].get('lower_multiplier', -np.inf) * std
            # upper_bound = mean + specs[param].get('upper_multiplier', np.inf) * std
            # samples = np.clip(samples, lower_bound, upper_bound)
            
            sampled_params.append(samples)
        else:
            # If parameter isn't being sampled, set to its mean value
            sampled_params.append(np.full((num_samples,), specs[param]['mean']))

    # Convert to numpy array for easier manipulation
    samples_array = np.array(sampled_params).T
    
    return samples_array, params_to_sample

def pdf_value(values, norm_info):
    """Input:
    - values:  (N,5) array of not normalized parameter values
    - specs:   dict with specs for each parameter"""
    """Output:
    - pdf:  (N,) array of pdf values for each combination of parameters"""
    norm_info_cop = norm_info.copy()
    norm_info_cop.pop('t', None)
    pdf = np.ones(values.shape[0])
    for i, param in enumerate(norm_info_cop.keys()):
        mean = norm_info_cop[param]['mean']
        std = norm_info_cop[param]['std']
        pdf *= norm.pdf(values[:, i], loc=mean, scale=std)
    return pdf

def initial_MCMC_samples(specs, norm_info, num_initial_samples=10):
    lhs_sampler = qmc.LatinHypercube(len(norm_info)-1)  # -1 because 't' is not sampled
    initial_lhs = lhs_sampler.random(num_initial_samples)
    initial_samples= np.zeros((num_initial_samples, 5))
    j=0
    for i, param in enumerate([key for key in specs.keys() if key != 't']):
        if param in norm_info:
            mean = specs[param]['mean']
            std = specs[param]['std']
            lower_bound = mean + specs[param]['lower_multiplier'] * std
            upper_bound = mean + specs[param]['upper_multiplier'] * std
            initial_samples[:, i] = initial_lhs[:, j] * (upper_bound - lower_bound) + lower_bound
            j += 1
        else:
            # If the parameter is not in the specs, just use its mean value
            initial_samples[:, i] = np.full((num_initial_samples,), specs[param]['mean'])

    return(initial_samples)

def metropolis_hastings(PINN_model, t_coll, num_samples, burn_in, num_initial_samples, norm_info, specs, sigma_hat_e=None, proposal_scale = 1):

    u_curr = initial_MCMC_samples(specs, norm_info, num_initial_samples)
    u_prop = np.copy(u_curr)
    accepted_count = 0
    t_batch=t_coll.view(1, t_coll.shape[0], 1).expand(u_prop.shape[0], t_coll.shape[0], 1)  # (B, Nt, 1)
    if sigma_hat_e is None:
        sigma_hat_e = 0.01 * np.std(u_curr)   # Initial guess - must be improved later 
    accepted_samples = np.zeros((num_samples, num_initial_samples, u_curr.shape[1]))

    for i in range(num_samples+burn_in):
        # Generate a new sample from the proposal distribution
        for j, param in enumerate(['m', 'mu', 'k', 'y0', 'v0']):
            if param in norm_info:
                u_prop[:, j] = np.random.normal(loc=u_curr[:, j], scale=proposal_scale*specs[param]['std'], size=u_curr.shape[0])

        # Evaluate the limit state function G(x) for the proposal sample
        input_to_g_prop = make_input_params_batched(t_coll, torch.from_numpy(u_prop).float(), norm_info)
        
        g_hat_prop = limit_state_function_G_batched(
            PINN_model,
            t_batch,
            input_to_g_prop
        ).detach() # detach() to avoid gradients = True 

        # Evaluate the limit state function G(x) for the current sample
        input_to_G_curr = make_input_params_batched(t_coll, torch.from_numpy(u_curr).float(), norm_info)

        g_hat_curr = limit_state_function_G_batched(
            PINN_model,
            t_batch,
            input_to_G_curr
        ).detach() # detach() to avoid gradients = True

        # Evaluate the pdf values for the current and proposal samples
        pdf_curr = pdf_value(u_curr, norm_info)
        pdf_prop = pdf_value(u_prop, norm_info)

        # Evaluate the pi-function for the proposal and current samples
        pi_prop= norm.cdf(-g_hat_prop/sigma_hat_e)
        pi_curr= norm.cdf(-g_hat_curr/sigma_hat_e)

        # Evaluate the h-function for the proposal and current samples
        h_prop = pi_prop * pdf_prop
        h_curr = pi_curr * pdf_curr

        alpha = h_prop / h_curr

        # Accept or reject the proposal based on the acceptance ratio
        rand = np.random.rand(num_initial_samples)
        accept = alpha > rand
        u_curr = np.where(accept[:, None], u_prop, u_curr)

        if i >= burn_in:
            accepted_count += np.sum(accept) # total number of accepted proposals, not stored samples
            accepted_samples[i - burn_in] = u_curr            

    return accepted_samples, accepted_count

from sklearn.cluster import KMeans

def select_cluster_centers(accepted_samples, n_clusters=30, random_state=0):
    """
    Flatten accepted MCMC samples and apply k-means to select Nc points.

    Parameters:
        accepted_samples (np.ndarray): shape (num_samples, num_chains, n_params)
        n_clusters (int): number of clusters (default 30)
        random_state (int): for reproducibility

    Returns:
        centers (np.ndarray): shape (n_clusters, n_params)
    """
    # Flatten the samples: shape (num_samples * num_chains, n_params)
    flattened_samples = accepted_samples.reshape(-1, accepted_samples.shape[-1])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(flattened_samples)

    # Extract the cluster centers
    centers = kmeans.cluster_centers_

    return centers

# Evaluate the exact solution and PINN_model prediction for k_means_centers
def compare_g_exact_and_pred(PINN_model, t_coll, t_test, datapoints, norm_info):
    """
    Compare the exact solution and the PINN model prediction for k-means centers.

    Output:
    - compare_g:  (N,3) array of g_exact, g_pred, and g_exact - g_pred
    - kmean_all_y_exact:  (N, Nt) array of y_exact for each k-means center
    """
    kmean_tensor = torch.from_numpy(datapoints).float()

    kmean_input = make_input_params_batched(t_coll, kmean_tensor, norm_info)

    t_batch = t_coll.view(1, t_coll.shape[0], 1).expand(kmean_tensor.shape[0], t_coll.shape[0], 1)  # (B, Nt, 1)
    
    kmean_g_pred = limit_state_function_G_batched(
        PINN_model,
        t_batch,
        kmean_input
    ).detach().numpy().tolist()

    kmean_g_exact = np.empty(datapoints.shape[0])
    kmean_all_y_exact = np.empty((datapoints.shape[0],t_test.shape[0]))
    for i in range(datapoints.shape[0]):
        y_exact = damped_harmonic_oscillator(t_test, *datapoints[i])
        g_exact = y_exact.min() + 1.0
        kmean_g_exact[i] = g_exact.item()
        kmean_all_y_exact[i] = y_exact
    # Convert lists to NumPy arrays for element-wise subtraction
    k_mean_g_exact_array = np.array(kmean_g_exact)
    k_mean_g_pred_array = np.array(kmean_g_pred)

    compare_g = np.column_stack((k_mean_g_exact_array, k_mean_g_pred_array, k_mean_g_exact_array - k_mean_g_pred_array))
    return(compare_g, kmean_all_y_exact)