import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import math
import os


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# # Define the PINN model
# class PINN_vanilla_oscillator(nn.Module):
#     def __init__(self, hidden_size=20, hidden_layers=3):
#         super(PINN_vanilla_oscillator, self).__init__()
#         input_dim = 6
#         layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]
        
#         for _ in range(hidden_layers):
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(nn.Tanh())
        
#         layers.append(nn.Linear(hidden_size, 1))
#         self.net = nn.Sequential(*layers)
    
#     def forward(self, t, m, mu, k, y0, v0):
#         x = torch.cat([t, m, mu, k, y0, v0], dim=1)
#         return self.net(x)

class PINN_vanilla_oscillator(nn.Module):
    def __init__(self, hidden_size=20, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(6, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, m, mu, k, y0, v0):
        # t, m, mu, k, y0, v0 each have shape (..., 1)
        orig_shape = t.shape        # e.g. (Nt,1) or (B,Nt,1)
        # compute batch_size = product of all dims except the last one
        batch_size = math.prod(orig_shape[:-1])  # pure-Python, no .item()

        # flatten all leading dims into one
        def flatten(x):
            return x.reshape(batch_size, -1)  # -1 is the feature dim (1)

        t_flat  = flatten(t)
        m_flat  = flatten(m)
        mu_flat = flatten(mu)
        k_flat  = flatten(k)
        y0_flat = flatten(y0)
        v0_flat = flatten(v0)
        
        # concatenate features -> (batch_size, 6)
        x_flat = torch.cat([t_flat, m_flat, mu_flat, k_flat, y0_flat, v0_flat], dim=1)
        
        # single pass through the MLP
        y_flat = self.net(x_flat)       # (batch_size, 1)
        
        # reshape back to original (..., 1)
        return y_flat.view(*orig_shape)

# class P2INN_oscillator(nn.Module):
#     def __init__(self,
#                  # parameter encoder settings
#                  param_hidden: int = 32,
#                  param_layers: int = 4,
#                  # coordinate encoder settings
#                  coord_hidden: int = 16,
#                  coord_layers: int = 3,
#                  # decoder / manifold net settings
#                  decoder_hidden: int = 64,
#                  decoder_layers: int = 6):
#         super().__init__()

#         # 1) build parameter encoder with param_layers of (Linear→Tanh)
#         pe_layers = []
#         # first layer: param_in → param_hidden
#         pe_layers += [nn.Linear(5,  param_hidden), nn.Tanh()]
#         # then (param_layers-1) more hidden (param_hidden → param_hidden)
#         for _ in range(param_layers - 1):
#             pe_layers += [nn.Linear(param_hidden, param_hidden), nn.Tanh()]
#         self.param_encoder = nn.Sequential(*pe_layers)

#         # 2) coordinate encoder with coord_layers of (Linear→Tanh)
#         ce_layers = [ nn.Linear(1, coord_hidden), nn.Tanh() ]
#         for _ in range(coord_layers - 1):
#             ce_layers += [nn.Linear(coord_hidden, coord_hidden), nn.Tanh()]
#         self.coord_encoder = nn.Sequential(*ce_layers)

#         # 3) decoder: input dim = param_hidden + coord_hidden
#         dec_layers = [
#             nn.Linear(param_hidden + coord_hidden, decoder_hidden),
#             nn.Tanh()
#         ]
#         for _ in range(decoder_layers - 1):
#             dec_layers += [ nn.Linear(decoder_hidden, decoder_hidden), nn.Tanh() ]
#         dec_layers.append(nn.Linear(decoder_hidden, 1))
#         self.decoder = nn.Sequential(*dec_layers)
    # def forward(self, t, m, mu, k, y0, v0):
    #     # form the two inputs
    #     eq_input = torch.cat([m, mu, k, y0, v0], dim=1)  # (batch,5)
    #     tc_input = t                                      # (batch,1)

    #     h_p = self.param_encoder(eq_input)                # (batch,param_hidden)
    #     h_c = self.coord_encoder(tc_input)                # (batch,coord_hidden)
    #     h   = torch.cat([h_p, h_c], dim=1)                # (batch,param_hidden+coord_hidden)
    #     y   = self.decoder(h)                             # (batch,1)
    #     return y

class P2INN_oscillator(nn.Module):
    def __init__(self,
                 # parameter encoder settings
                 param_hidden: int = 32,
                 param_layers: int = 4,
                 # coordinate encoder settings
                 coord_hidden: int = 16,
                 coord_layers: int = 3,
                 # decoder / manifold net settings
                 decoder_hidden: int = 64,
                 decoder_layers: int = 6):
        super().__init__()

        # 1) Parameter encoder
        pe_layers = [nn.Linear(5, param_hidden), nn.Tanh()]
        for _ in range(param_layers - 1):
            pe_layers += [nn.Linear(param_hidden, param_hidden), nn.Tanh()]
        self.param_encoder = nn.Sequential(*pe_layers)

        # 2) Coordinate encoder
        ce_layers = [nn.Linear(1, coord_hidden), nn.Tanh()]
        for _ in range(coord_layers - 1):
            ce_layers += [nn.Linear(coord_hidden, coord_hidden), nn.Tanh()]
        self.coord_encoder = nn.Sequential(*ce_layers)

        # 3) Decoder
        dec_layers = [nn.Linear(param_hidden + coord_hidden, decoder_hidden), nn.Tanh()]
        for _ in range(decoder_layers - 1):
            dec_layers += [nn.Linear(decoder_hidden, decoder_hidden), nn.Tanh()]
        dec_layers.append(nn.Linear(decoder_hidden, 1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, t, m, mu, k, y0, v0):
        # t, m, mu, k, y0, v0 each have shape (..., 1)
        orig_shape = t.shape                  # e.g. (Nt,1) or (B,Nt,1)
        batch_size = math.prod(orig_shape[:-1])

        # flatten helper: (...,1) -> (batch_size,1)
        def flatten(x):
            return x.reshape(batch_size, -1)

        t_flat  = flatten(t)
        m_flat  = flatten(m)
        mu_flat = flatten(mu)
        k_flat  = flatten(k)
        y0_flat = flatten(y0)
        v0_flat = flatten(v0)

        # 1) parameter encoding
        eq_input = torch.cat([m_flat, mu_flat, k_flat, y0_flat, v0_flat], dim=1)  
        h_p = self.param_encoder(eq_input)            # (batch_size, param_hidden)

        # 2) coordinate encoding
        h_c = self.coord_encoder(t_flat)              # (batch_size, coord_hidden)

        # 3) decoding
        h      = torch.cat([h_p, h_c], dim=1)         # (batch_size, param_hidden+coord_hidden)
        y_flat = self.decoder(h)                      # (batch_size, 1)

        # 4) reshape back to (...,1)
        return y_flat.view(*orig_shape)

def z_score_denormalize(x, norm_info_param):
    """
    Inverts a Z-score normalization.
    Args:
        x (torch.Tensor): The normalized tensor.
        norm_info_param (dict): Dictionary with keys 'mean' and 'std' for this parameter.
    Returns:
        torch.Tensor: The tensor transformed back to physical space.
    """
    return x * norm_info_param['std'] + norm_info_param['mean']

def z_score_normalize(x, norm_info_param):
    """
    Applies Z-score normalization.
    Args:
        x (torch.Tensor): The tensor to normalize.
        norm_info_param (dict): Dictionary with keys 'mean' and 'std' for this parameter.
    Returns:
        torch.Tensor: The normalized tensor.
    """
    return (x - norm_info_param['mean']) / norm_info_param['std']

def pde_loss(model, t, m, mu, k, y0_val, v0_val, norm_info=None):
    # First, compute the model output using the (normalized) inputs.
    y = model(t, m, mu, k, y0_val, v0_val)
    y_t_norm = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    y_tt_norm = torch.autograd.grad(y_t_norm, t, grad_outputs=torch.ones_like(y_t_norm), create_graph=True)[0]

    # If norm_info is provided, denormalize the parameters for the PDE residual.
    if norm_info is not None:
        dt_dt_hat = norm_info['t']['range'] #Scaling the time collocation points to the physical time range using chain rule.
        y_t = y_t_norm / dt_dt_hat
        y_tt = y_tt_norm  / (dt_dt_hat**2)

        m_phys = z_score_denormalize(m, norm_info['m']) if 'm' in norm_info else m.detach()
        mu_phys = z_score_denormalize(mu, norm_info['mu']) if 'mu' in norm_info else mu.detach()
        k_phys = z_score_denormalize(k, norm_info['k']) if 'k' in norm_info else k.detach()
    else:
        y_t = y_t_norm
        y_tt = y_tt_norm

        m_phys = m
        mu_phys = mu
        k_phys = k

    # Compute the PDE residual using physical parameters.
    residual = m_phys * y_tt + mu_phys * y_t + k_phys * y
    return torch.mean(residual**2)

# Compute boundary loss
def boundary_loss(model, t0, m, mu, k, y0, v0,norm_info=None):
    y_pred = model(t0, m, mu, k, y0, v0)
    y_t_norm = torch.autograd.grad(y_pred, t0, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]

    if norm_info is not None:
        dt_dt_hat = norm_info['t']['range'] #Scaling the time collocation points to the physical time range using chain rule.
        y_t = y_t_norm / dt_dt_hat
        y0_phys = z_score_denormalize(y0, norm_info['y0']) if 'y0' in norm_info else y0.detach()
        v0_phys = z_score_denormalize(v0, norm_info['v0']) if 'v0' in norm_info else v0.detach()
    else:
        y_t = y_t_norm
        y0_phys = y0
        v0_phys = v0
    # Compute the boundary loss using physical parameters.
    return torch.mean((y_pred - y0_phys)**2 + (y_t - v0_phys)**2)

from batching import make_input_params_batched
# Include the evasluation of exact solutions into the loss 
def data_loss(PINN_model, datapoints, y_exact_datapoints, norm_info, lambda_data=1.0):
    """
    Compute the data loss for the PINN model.

    Args:
        PINN_model: The PINN model.
        datapoints: The centers of the clusters.
        y_exact_datapoints: The exact solutions at the cluster centers. (numpy)
        lambda_data: The weight for the data loss.

    Returns:
        The data loss value scaled by lambda_data.
    """
    timepoints = torch.linspace(0, 1, len(y_exact_datapoints[0])).view(-1, 1)

    # Compute the predicted values at the cluster centers
    pred_params = make_input_params_batched(timepoints, torch.from_numpy(datapoints).float(), norm_info)

    t_batch = timepoints.view(1, timepoints.shape[0], 1).expand(y_exact_datapoints.shape[0], timepoints.shape[0], 1)  # (B, Nt, 1)

    y_pred = PINN_model(t_batch, *pred_params).squeeze(-1)  # (B, Nt, 1) -> (B, Nt)

    # Reshape y_exact_datapoints to match y_pred
    y_exact_tensor = torch.from_numpy(y_exact_datapoints).float()

    # Compute the data loss
    data_loss_value = torch.mean((y_pred - y_exact_tensor)**2)
    return data_loss_value * lambda_data  

def plot_loss(losses_dict):
    plt.figure(figsize=(5, 3))
    # Find the maximum length among non-empty loss lists
    total_epochs = max((len(v) for v in losses_dict.values()), default=0)

    for loss_name, loss_values in losses_dict.items():
        n = len(loss_values)
        if n == 0:
            continue  # skip empty losses

        # If this curve is shorter than the longest one, offset it to the right
        if n < total_epochs:
            offset = total_epochs - n
            x = range(offset, offset + n)
        else:
            x = range(n)

        plt.plot(x, loss_values, label=loss_name)

    plt.yscale('log')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Trainer class to manage training process
class Trainer:
    def __init__(self, model, optimizer, epochs=4001, lambda_residual = 1.0, lambda_bc=10.0, lambda_data=1.0, datapoints = None, y_exact_datapoints=None):

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        if os.path.exists("loss_history.npy"): 
            self.losses = np.load("loss_history.npy", allow_pickle=True).tolist()
        else:
            self.losses = {"Residual Loss": [], "Boundary Loss": [], "Data Loss": []}
        self.lambda_residual = lambda_residual
        self.lambda_bc = lambda_bc
        self.lambda_data = lambda_data
        self.datapoints = datapoints
        self.y_exact_datapoints = y_exact_datapoints

    def train(self, *args):
        # Allow passing a single dictionary argument, or individual tensors.
        if args and isinstance(args[0], dict):
            params = args[0]
            t_coll = params['t_coll']
            m_val  = params['m']
            mu_val = params['mu']
            k_val  = params['k']
            y0_val = params['y0']
            v0_val = params['v0']
            t0     = params['t0']
            # define the norm_info dictionary if it exists in params
            norm_info = params.get('norm_info', None)
        else:
            t_coll, m_val, mu_val, k_val, y0_val, v0_val, t0 = args
            norm_info = None

        # Initialize best loss so far
        # if all three lists are empty, use +∞
        if not any(self.losses.values()):
            best_loss = float("inf")
        else:
            # otherwise grab last entry or 0.0
            res_last  = self.losses["Residual Loss"][-1]  if self.losses["Residual Loss"] else 0.0
            bc_last   = self.losses["Boundary Loss"][-1]  if self.losses["Boundary Loss"] else 0.0
            data_last = self.losses["Data Loss"][-1]      if self.losses["Data Loss"] else 0.0

            best_loss = (
                res_last  * self.lambda_residual +
                bc_last   * self.lambda_bc +
                data_last * self.lambda_data
            )

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss_pde = pde_loss(self.model, t_coll, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss_bc = boundary_loss(self.model, t0, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            self.losses["Residual Loss"].append(loss_pde.item())
            self.losses["Boundary Loss"].append(loss_bc.item())

            # If data loss is provided, add it to the total loss
            if self.datapoints is not None and self.y_exact_datapoints is not None:
                loss_data = data_loss(self.model, self.datapoints, self.y_exact_datapoints, norm_info, lambda_data=self.lambda_data)
                self.losses["Data Loss"].append(loss_data.item())
                loss = loss_pde * self.lambda_residual + self.lambda_bc * loss_bc + loss_data * self.lambda_data
            else:
                # If no data loss is provided, just use the PDE and boundary losses
                loss = loss_pde * self.lambda_residual + loss_bc * self.lambda_bc
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, PDE loss: {loss_pde.item()}, BC loss: {loss_bc.item()}")
                plot_loss(self.losses)
                np.save("loss_history.npy", self.losses)
        print(f"Phase 1 complete. Best loss so far: {best_loss}")

        # Phase 2: Continue training until an epoch is reached with loss below the best_loss from Phase 1.
        extra_epochs = 0
        while True:
            self.optimizer.zero_grad()
            loss_pde = pde_loss(self.model, t_coll, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss_bc = boundary_loss(self.model, t0, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            self.losses["Residual Loss"].append(loss_pde.item())
            self.losses["Boundary Loss"].append(loss_bc.item())

            # If data loss is provided, add it to the total loss
            if self.datapoints is not None and self.y_exact_datapoints is not None:
                loss_data = data_loss(self.model, self.datapoints, self.y_exact_datapoints, norm_info, lambda_data=self.lambda_data)
                self.losses["Data Loss"].append(loss_data.item())
                loss = loss_pde * self.lambda_residual + self.lambda_bc * loss_bc + loss_data * self.lambda_data
            else:
                # If no data loss is provided, just use the PDE and boundary losses
                loss = loss_pde * self.lambda_residual + loss_bc * self.lambda_bc

            loss.backward()
            self.optimizer.step()

            extra_epochs += 1
            epoch += 1
            current_loss = loss.item()

            if extra_epochs % 1000 == 0:
                print(f"Extra Epoch {extra_epochs}, PDE loss: {loss_pde.item()}, BC loss: {loss_bc.item()}")
                plot_loss(self.losses)
                np.save("loss_history.npy", self.losses)
            if current_loss < best_loss:
                print(f"Improved loss found: {current_loss} (after {extra_epochs} extra epochs)")
                plot_loss(self.losses)
                np.save("loss_history.npy", self.losses)
                break