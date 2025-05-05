import numpy as np
import torch
from utils import z_score_normalize, z_score_denormalize

def make_input_params_batched(t_coll, values, norm_info=None):
    """
    Makes batched test parameters for input to the PINN model.

    Args:
      t_test:  torch.Tensor of shape (Nt,) or array-like
      values:  torch.Tensor shape (5,), or (B,5)
      norm_info:  optional dict for z-score normalization
      device:     torch.device (defaults to t_test.device)

    Returns:
      Tuple of 5 torch.Tensors, each of shape (B, Nt, 1).
    """
    Nt = t_coll.shape[0]

    # If it’s 1D-length-5, make it shape (1,5)
    if values.dim() == 1:
        values = values.unsqueeze(0)

    B = values.shape[0]
    if values.shape[1] != 5:
        raise ValueError(f"Expected second dim of size 5, got {values.shape}")

    for i, param in enumerate(['m', 'mu', 'k', 'y0', 'v0']):
        # 3) (Optional) apply normalization in batch
        if norm_info and param in norm_info:
            # Example for each column; adapt your z_score_normalize to accept Tensors
            values[:, i] = z_score_normalize(values[:, i], norm_info[param])

    # 4) Expand each scalar to (B, Nt, 1)
    # values[:,i] is (B,)
    m  = values[:, 0][:, None, None].expand(B, Nt, 1)
    mu = values[:, 1][:, None, None].expand(B, Nt, 1)
    k  = values[:, 2][:, None, None].expand(B, Nt, 1)
    y0 = values[:, 3][:, None, None].expand(B, Nt, 1)
    v0 = values[:, 4][:, None, None].expand(B, Nt, 1)

    return m, mu, k, y0, v0

def limit_state_function_G_batched(
    model,
    t_batch,                # torch.Tensor, shape (Nt,)
    pred_params,      # tuple of (m, mu, k, y0, v0), each shape (B, Nt, 1)
):
    """
    Batched limit‐state: G = min_t y_pred + 1.0
    Returns a torch.Tensor of shape (B,)
    """
    # Unpack
    m, mu, k, y0, v0 = pred_params
    # Call the PINN once over the whole batch
    # y_pred: (B, Nt, 1)
    y_pred = model(t_batch, m, mu, k, y0, v0)
    # Squeeze off the last dim and min over time-axis
    # y_pred.squeeze(-1): (B, Nt)
    min_y, _ = torch.min(y_pred.squeeze(-1), dim=1)  # returns (values, indices)

    # G = min_y + 1.0  → shape (B,)
    return min_y + 1.0

def pde_loss_batched(model,
                     t, m, mu, k, y0_val, v0_val,
                     norm_info=None):
    """
    Batched PDE residual loss:
      residual = m*y_tt + mu*y_t + k*y
    Returns:
      mse_per_sample: torch.Tensor, shape (B,)
    """
    # 1) Forward pass

    y_pred = model(t, m, mu, k, y0_val, v0_val)       # (B, Nt, 1)
    # 2) Time-derivatives via autograd
    grad_ones = torch.ones_like(y_pred)
    y_t_norm  = torch.autograd.grad(y_pred, t, grad_outputs=grad_ones,
                                    create_graph=True)[0]
    y_tt_norm = torch.autograd.grad(y_t_norm, t, grad_outputs=grad_ones,
                                    create_graph=True)[0]

    # 3) Chain-rule if normalized time
    if norm_info is not None:
        dt_scale = norm_info['t']['range']
        y_t  = y_t_norm  / dt_scale
        y_tt = y_tt_norm / (dt_scale**2)
    else:
        y_t, y_tt = y_t_norm, y_tt_norm

    # 4) Denormalize parameters (we grab the first time-slice since params are constant in t)
    #    and reshape back so we can broadcast
    def denorm_and_expand(param, key):
        # param: (B, Nt, 1) → take param[:,0,0] → denorm → (B,) → (B,1,1)
        vals = param[:, 0, 0]
        if norm_info is not None and key in norm_info:
            vals = z_score_denormalize(vals, norm_info[key])
        return vals.view(-1, 1, 1)

    m_phys  = denorm_and_expand(m,  'm')
    mu_phys = denorm_and_expand(mu, 'mu')
    k_phys  = denorm_and_expand(k,  'k')

    # 5) Compute residual and MSE per sample
    residual = m_phys * y_tt + mu_phys * y_t + k_phys * y_pred  # (B, Nt,1)
    mse_per_sample = (residual**2).mean(dim=(1,2))               # (B,)
    return mse_per_sample

def boundary_loss_batched(model,
                          t0,          # torch.Tensor, shape (B, 1, 1), requires_grad=True
                          m, mu, k,
                          y0, v0,
                          norm_info=None):
    """
    Batched boundary loss at t=0:
      loss = (y_pred - y0_phys)^2 + (y_t - v0_phys)^2
    Returns:
      mse_per_sample: torch.Tensor, shape (B,)
    """
    # 1) Forward & derivative at initial time
    y_pred = model(t0, m, mu, k, y0, v0)                   # (B,1,1)
    grad_ones = torch.ones_like(y_pred)
    y_t_norm = torch.autograd.grad(y_pred, t0,
                                   grad_outputs=grad_ones,
                                   create_graph=True)[0]       # (B,1,1)

    # 2) Chain-rule if normalized time
    if norm_info is not None:
        dt_scale = norm_info['t']['range']
        y_t = y_t_norm / dt_scale
    else:
        y_t = y_t_norm

    # 3) Denormalize y0 and v0 (they’re constant over t0)
    y0_phys = y0[:, 0, 0]
    v0_phys = v0[:, 0, 0]
    if norm_info is not None:
        y0_phys = z_score_denormalize(y0_phys, norm_info['y0']) if 'y0' in norm_info else y0_phys
        v0_phys = z_score_denormalize(v0_phys, norm_info['v0']) if 'v0' in norm_info else v0_phys

    # 4) reshape to broadcast back to (B,1,1)
    y0_phys = y0_phys.view(-1,1,1)
    v0_phys = v0_phys.view(-1,1,1)

    # 5) compute per‐sample squared error and mean
    err = (y_pred - y0_phys)**2 + (y_t - v0_phys)**2  # (B,1,1)
    mse_per_sample = err.mean(dim=(1,2))              # (B,)

    return mse_per_sample

def evaluate_G_and_residual_batchwise(
    PINN_model,
    samples,        # (N,5) float32 tensor
    t_coll,         # (Nt,) float32 tensor
    norm_info,         # your dict with norm_info
    lambda_bc=2.0,
    batch_size=64):
    
    Nt = t_coll.shape[0]
    N = samples.shape[0]
    G_outs = np.empty(N, dtype=np.float32)
    R_outs = np.empty(N, dtype=np.float32)
    
    t0 = torch.zeros_like(t_coll)
    

    for start in range(0, N, batch_size):
        end = min(start + batch_size,N)
        batch_params = samples[start:end] # (B,5)

        B_cur=batch_params.shape[0]

        t_batch=t_coll.view(1, Nt, 1).expand(B_cur, Nt, 1)  # (B, Nt, 1)
        t0_batch=t0.view(1, Nt, 1).expand(B_cur, Nt, 1)  # (B, Nt, 1)
        
        # ---- limit_state_function_G batching ----
        # split into per-parameter tensors
        # m_b, mu_b, k_b, y0_b, v0_b = torch.unbind(batch_params, dim=1)
        
        # make_input_params must handle 1D-per-param -> (B, Nt, 1) or similar
        # Here we assume it will repeat each scalar over t_coll internally
        input_params = make_input_params_batched(
            t_coll,      # the same t_coll
            batch_params,
            norm_info
        )

        # G_vals: returns (B,) tensor
        with torch.no_grad():
            Gb = limit_state_function_G_batched(
                PINN_model,
                t_batch,
                input_params, 
            )  # -> shape (B,)

        # ---- PDE & boundary residuals batching ----

        pde_b = pde_loss_batched(
            PINN_model,
            t_batch.requires_grad_(True),
            *input_params,
            norm_info
        ) 
        
        bc_b = boundary_loss_batched(
            PINN_model,
            t0_batch.requires_grad_(True),
            *input_params,
            norm_info
        )
        
        Rb = pde_b + lambda_bc * bc_b
        
        G_outs[start:end] = Gb.numpy()
        R_outs[start:end] = Rb.detach().numpy()

    return G_outs, R_outs

def evaluate_G_batchwise(
    PINN_model,
    samples,        # (N,5) float32 tensor
    t_coll,         # (Nt,) float32 tensor
    norm_info,         # your dict with norm_info
    batch_size=64):
    
    Nt = t_coll.shape[0]
    N = samples.shape[0]
    G_outs = np.empty(N, dtype=np.float32)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size,N)
        batch_params = samples[start:end] # (B,5)

        B_cur=batch_params.shape[0]

        t_batch=t_coll.view(1, Nt, 1).expand(B_cur, Nt, 1)  # (B, Nt, 1)
        
        # ---- limit_state_function_G batching ----
        # split into per-parameter tensors
        # m_b, mu_b, k_b, y0_b, v0_b = torch.unbind(batch_params, dim=1)
        
        # make_input_params must handle 1D-per-param -> (B, Nt, 1) or similar
        # Here we assume it will repeat each scalar over t_coll internally
        input_params = make_input_params_batched(
            t_coll,      # the same t_coll
            batch_params,
            norm_info
        )

        # G_vals: returns (B,) tensor
        with torch.no_grad():
            Gb = limit_state_function_G_batched(
                PINN_model,
                t_batch,
                input_params, 
            )  # -> shape (B,)

        G_outs[start:end] = Gb.numpy()
    
    return G_outs