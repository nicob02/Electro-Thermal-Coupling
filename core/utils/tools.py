import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import math
from torch_geometric.data import Data
import torch.autograd as autograd

def RemoveDir(filepath):
    '''
    If the folder doesn't exist, create it; and if it exists, clear it.
    '''
    if not os.path.exists(filepath):
        os.makedirs(filepath,exist_ok=True)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)


class Config:
    def __init__(self) -> None:
        pass
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


def parse_config(file='config.json'):
    configs = Config() 
    if not os.path.exists(file):
        return configs
    with open(file, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            config = Config()
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    config.setattr(k1, v1)
            else:
                raise TypeError
            configs[k] = config
    return configs[k]

'''
def modelTrainer(config):
    
    model = config.model
    graph = config.graph
    scheduler = torch.optim.lr_scheduler.StepLR(
        config.optimizer, step_size=config.lrstep, gamma=0.99)  
    best_loss  = np.inf
    func  = config.func_main
    opt   = config.optimizer
    tol = 1e-4
    x_coord = graph.pos[:, 0:1]               # shape [N,1]
    is_left  = torch.isclose(x_coord, torch.zeros_like(x_coord), atol=tol)
    is_right = torch.isclose(x_coord, torch.ones_like(x_coord),  atol=tol)
    lateral_mask = (is_left | is_right).squeeze()   
    ramp_steps = config.change_sigma_epoch 
    lr_drop_epoch = ramp_steps + 1
    lr_drop_factor = 0.1
    # 1) Build static node features once
    graph = func.graph_modify(graph)
    
    for epoch in range(1, config.epchoes + 1):  # Creates different ic and solves the problem, does this epoch # of times

        t = min(epoch, ramp_steps)
        sigma_val = 1.0 + 3.0 * (t / ramp_steps)
        func.sigma = torch.ones(graph.num_nodes,1,
                                device=graph.pos.device) * sigma_val
         rebuild node features so graph.x picks up new σ
 
        graph = func.graph_modify(graph)
            
        raw = model(graph)                     # [N,2] raw outputs
        PV, PT, grad_V = func.pde_residuals(graph, raw)  # both [N,1] 
        
        loss_int = torch.mean(PV**2) + 10*torch.mean(PT**2)
        du_dx     = grad_V[:, 0:1]        # partial derivative of voltage with respect to normal(x-direction)
        du_dx_lat = du_dx[lateral_mask]
        loss_neu  = torch.norm(du_dx_lat)**2 / du_dx_lat.numel()

        loss = loss_int + 2000*loss_neu
        config.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        config.optimizer.step()
        scheduler.step()

         # 5) at end of ramp, drop LR by 10× once
        if epoch == lr_drop_epoch:
            for pg in opt.param_groups:
                pg['lr'] *= lr_drop_factor
            print(f"→ [Epoch {epoch}] σ reached {sigma_val:.3f}; "
                  f"dropped LR to {opt.param_groups[0]['lr']:.1e}")
        if epoch % 500 == 0:
            print(f"[Epoch {epoch:4d}] Loss = {loss.item():.3e}")
            
    model.save_model(config.optimizer)
    print('model saved at loss: %.4e' % loss)    
    print("Training completed!")
'''
def modelTrainer(config):
    model = config.model
    graph = config.graph
    func  = config.func_main
    opt   = config.optimizer

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=config.lrstep, gamma=0.99
    )

    ramp_epoch = config.change_domain_epoch
    step_size  = config.domain_step

    # unpack helper for lb/ru
    def _unpack(lb_tensor, ru_tensor):
        lb = lb_tensor.detach().view(-1)
        ru = ru_tensor.detach().view(-1)
        return float(lb[0]), float(lb[1]), float(ru[0]), float(ru[1])

    # store initial and final lb/ru as 1-D tensors
    init_lb = torch.tensor(func.lb, device=graph.pos.device).view(-1)
    init_ru = torch.tensor(func.ru, device=graph.pos.device).view(-1)
    final_lb = torch.tensor(config.new_lb, device=graph.pos.device).view(-1)
    final_ru = torch.tensor(config.new_ru, device=graph.pos.device).view(-1)

    # build features once (uses initial domain)
    graph = func.graph_modify(graph)

    # prepare Neumann mask (assumes x in [0,1] after normalization inside func if needed)
    tol = 1e-4
    x_coord   = graph.pos[:,0:1]
    is_left   = torch.isclose(x_coord, torch.zeros_like(x_coord), atol=tol)
    is_right  = torch.isclose(x_coord, torch.ones_like(x_coord),  atol=tol)
    lateral_mask = (is_left | is_right).squeeze()

    for epoch in range(1, config.epchoes + 1):
        # every `step_size` epochs (and while <= ramp_epoch), update domain
        if epoch % step_size == 0 and epoch <= ramp_epoch:
            α = epoch / ramp_epoch
            # linear interpolation
            new_lb = init_lb + (final_lb - init_lb) * α
            new_ru = init_ru + (final_ru - init_ru) * α
            func.lb = new_lb.clone()
            func.ru = new_ru.clone()
            lbx, lby, rux, ruy = _unpack(func.lb, func.ru)
            print(f"→ [Epoch {epoch:4d}] Domain now [{lbx:.2f},{rux:.2f}]×[{lby:.2f},{ruy:.2f}]")

        # rebuild node features (so graph.x can depend on domain if your func.graph_modify does)
        graph = func.graph_modify(graph)

        # forward + PDE residuals
        raw     = model(graph)
        PV, PT, grad_V = func.pde_residuals(graph, raw)

        # interior + Neumann losses
        loss_int = torch.mean(PV**2) + 10*torch.mean(PT**2)
        du_dx     = grad_V[:,0:1]
        du_dx_lat = du_dx[lateral_mask]
        loss_neu  = torch.norm(du_dx_lat)**2 / du_dx_lat.numel()

        loss = loss_int + 2000*loss_neu

        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"[Epoch {epoch:4d}] Loss = {loss.item():.3e}")

    model.save_model(config.optimizer)
    print('model saved at loss: %.4e' % loss.item())
    print("Training completed!")

@torch.no_grad()
def modelTester(config):
    """
    Single‐shot evaluation of the trained steady‐state GNN.
    Returns:
      V_pred, T_pred: NumPy arrays of shape [N,1]
    """
    # 1) Put model & graph on the right device
    model = config.model.to(config.device).eval()
    graph = config.graph.to(config.device)

    # 2) Build the static node features [x,y,σ,k]
    graph = config.func_main.graph_modify(graph)

    # 3) Forward + hard‐BC ansatz
    raw   = model(graph)  # [N,2] = [V_raw, T_raw]
    V_pred = config.func_main._ansatz_V(graph, raw[:,0:1])
    T_pred = config.func_main._ansatz_T(graph, raw[:,1:2])

    return V_pred.cpu().numpy(), T_pred.cpu().numpy()


def compute_steady_error(pred, exact):
    """
    Compute relative L2 error between pred and exact.
    Both pred & exact can be Tensor or NumPy of shape [N,1] or [N,].
    """
    # to NumPy flat
    if isinstance(pred, torch.Tensor):
        p = pred.detach().cpu().numpy().reshape(-1)
    else:
        p = np.array(pred).reshape(-1)
    if isinstance(exact, torch.Tensor):
        e = exact.detach().cpu().numpy().reshape(-1)
    else:
        e = np.array(exact).reshape(-1)

    num   = np.linalg.norm(p - e)
    denom = np.linalg.norm(e) + 1e-16
    return num/denom


def render_results(V_pred, T_pred, V_exact, T_exact, graph,
                   filename="steady_coupled.png"):
    """
    Plot Exact vs Predicted vs Absolute Error for both V and T.
    Saves a 2×3 panel figure.
    """
    pos = graph.pos.cpu().numpy()
    x, y = pos[:,0], pos[:,1]

    err_V = np.abs(V_exact - V_pred)
    err_T = np.abs(T_exact - T_pred)

    fig, axes = plt.subplots(2,3, figsize=(18,12))

    # Row 0: Voltage
    im0 = axes[0,0].scatter(x, y, c=V_exact.flatten(), s=5, cmap='viridis')
    axes[0,0].set_title("Exact Voltage")
    plt.colorbar(im0, ax=axes[0,0], shrink=0.7)

    im1 = axes[0,1].scatter(x, y, c=V_pred.flatten(),  s=5, cmap='viridis')
    axes[0,1].set_title("Predicted Voltage")
    plt.colorbar(im1, ax=axes[0,1], shrink=0.7)

    im2 = axes[0,2].scatter(x, y, c=err_V.flatten(),  s=5, cmap='magma')
    axes[0,2].set_title("Voltage Absolute Error")
    plt.colorbar(im2, ax=axes[0,2], shrink=0.7)

    # Row 1: Temperature
    im3 = axes[1,0].scatter(x, y, c=T_exact.flatten(), s=5, cmap='viridis')
    axes[1,0].set_title("Exact Temperature")
    plt.colorbar(im3, ax=axes[1,0], shrink=0.7)

    im4 = axes[1,1].scatter(x, y, c=T_pred.flatten(),  s=5, cmap='viridis')
    axes[1,1].set_title("Predicted Temperature")
    plt.colorbar(im4, ax=axes[1,1], shrink=0.7)

    im5 = axes[1,2].scatter(x, y, c=err_T.flatten(),  s=5, cmap='magma')
    axes[1,2].set_title("Temperature Absolute Error")
    plt.colorbar(im5, ax=axes[1,2], shrink=0.7)

    for ax in axes.flatten():
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
