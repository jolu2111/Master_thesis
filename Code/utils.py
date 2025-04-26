import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Define the PINN model
class PINN_vanilla_oscillator(nn.Module):
    def __init__(self, hidden_size=20, hidden_layers=3):
        super(PINN_vanilla_oscillator, self).__init__()
        input_dim = 6
        layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, m, mu, k, y0, v0):
        x = torch.cat([t, m, mu, k, y0, v0], dim=1)
        return self.net(x)

import torch
import torch.nn as nn

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

        # 1) build parameter encoder with param_layers of (Linear→Tanh)
        pe_layers = []
        # first layer: param_in → param_hidden
        pe_layers += [nn.Linear(5,  param_hidden), nn.Tanh()]
        # then (param_layers-1) more hidden (param_hidden → param_hidden)
        for _ in range(param_layers - 1):
            pe_layers += [nn.Linear(param_hidden, param_hidden), nn.Tanh()]
        self.param_encoder = nn.Sequential(*pe_layers)

        # 2) coordinate encoder with coord_layers of (Linear→Tanh)
        ce_layers = [ nn.Linear(1, coord_hidden), nn.Tanh() ]
        for _ in range(coord_layers - 1):
            ce_layers += [nn.Linear(coord_hidden, coord_hidden), nn.Tanh()]
        self.coord_encoder = nn.Sequential(*ce_layers)

        # 3) decoder: input dim = param_hidden + coord_hidden
        dec_layers = [
            nn.Linear(param_hidden + coord_hidden, decoder_hidden),
            nn.Tanh()
        ]
        for _ in range(decoder_layers - 1):
            dec_layers += [ nn.Linear(decoder_hidden, decoder_hidden), nn.Tanh() ]
        dec_layers.append(nn.Linear(decoder_hidden, 1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, t, m, mu, k, y0, v0):
        # form the two inputs
        eq_input = torch.cat([m, mu, k, y0, v0], dim=1)  # (batch,5)
        tc_input = t                                      # (batch,1)

        h_p = self.param_encoder(eq_input)                # (batch,param_hidden)
        h_c = self.coord_encoder(tc_input)                # (batch,coord_hidden)
        h   = torch.cat([h_p, h_c], dim=1)                # (batch,param_hidden+coord_hidden)
        y   = self.decoder(h)                             # (batch,1)
        return y


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

def plot_loss(epoch, losses_dict):
    plt.figure(figsize=(5, 3))
    for loss_name, loss_values in losses_dict.items():
        plt.plot(epoch, loss_values, label=loss_name)
    plt.yscale('log')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Trainer class to manage training process
class Trainer:
    def __init__(self, model, optimizer, epochs=4001, lambda_bc=10.0):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.losses = {"Residual Loss": [], "Boundary Loss": []}
        self.lambda_bc = lambda_bc

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

        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss_pde = pde_loss(self.model, t_coll, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss_bc = boundary_loss(self.model, t0, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss = loss_pde + self.lambda_bc * loss_bc
            loss.backward()
            self.optimizer.step()

            self.losses["Residual Loss"].append(loss_pde.item())
            self.losses["Boundary Loss"].append(loss_bc.item())
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, PDE loss: {loss_pde.item()}, BC loss: {loss_bc.item()}")
                plot_loss(range(epoch + 1), self.losses)
        print(f"Phase 1 complete. Best loss so far: {best_loss}")

        # Phase 2: Continue training until an epoch is reached with loss below the best_loss from Phase 1.
        extra_epochs = 0
        while True:
            self.optimizer.zero_grad()
            loss_pde = pde_loss(self.model, t_coll, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss_bc = boundary_loss(self.model, t0, m_val, mu_val, k_val, y0_val, v0_val, norm_info)
            loss = loss_pde + self.lambda_bc * loss_bc
            loss.backward()
            self.optimizer.step()

            extra_epochs += 1
            epoch += 1
            current_loss = loss.item()
            self.losses["Residual Loss"].append(loss_pde.item())
            self.losses["Boundary Loss"].append(loss_bc.item())

            if extra_epochs % 1000 == 0:
                print(f"Extra Epoch {extra_epochs}, PDE loss: {loss_pde.item()}, BC loss: {loss_bc.item()}")
                plot_loss(range(epoch + 1), self.losses)
            if current_loss < best_loss:
                print(f"Improved loss found: {current_loss} (after {extra_epochs} extra epochs)")
                plot_loss(range(epoch + 1), self.losses)
                break
