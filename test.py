import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import compute_steady_error, render_results
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import CoupledElectroThermalFunc as Func
import os
import matplotlib.pyplot as plt
import numpy as np
from FEM import run_fem 
from torch_geometric.data import Data

out_ndim = 2

dens=65
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)


mesh = ElectrodeMesh(ru=(0.5, 0.5), lb=(-0.5, -0.5), density=65)
graph = mesh.getGraphData()


sigma = torch.ones(graph.num_nodes,1,device=device) * 1.0
kappa = torch.ones(graph.num_nodes,1,device=device) * 1.0

# 3) Physics‐helper
func_main = Func(sigma=sigma, k=kappa, V_D=1.0, T_D=273.0)

model = msgPassing(message_passing_num=3, node_input_size=out_ndim+2, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20

test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
setattr(test_config, 'device', device)   
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref)
setattr(test_config, 'ndim', out_ndim)
setattr(test_config, 'graph_modify', func_main.graph_modify)
setattr(test_config, 'graph', graph)
setattr(test_config, 'density', dens)
setattr(test_config, 'func_main', func_main)
      

#-----------------------------------------

print('************* model test starts! ***********************')
V_pred, T_pred = modelTester(test_config)       # returns an NumPy array [N,2]
# 2) Run the FEM solver
coords_fem, V_vals_fem, T_vals_fem = run_fem(
    mesh=mesh,
    coords=graph.pos.cpu().numpy(),
    sigma_vals = sigma.cpu().numpy().flatten(),
    k_vals     = kappa.cpu().numpy().flatten(),
    lb         = (-0.5, -0.5),   # example for testing on [-1,1]^2
    ru         = ( 0.5,  0.5),
    V_D        = 1.0,            # your Dirichlet voltage top face
    T_D        = 273.0           # your Dirichlet temperature on all faces
)

# ensure ordering of coords matches GNN's graph.pos; if not, you'd need to reorder.

# 3) Compute and print relative L2 errors
err_V = compute_steady_error(V_pred, V_vals_fem)
err_T = compute_steady_error(T_pred, T_vals_fem)
print(f"Rel L2 error Voltage:     {err_V:.3e}")
print(f"Rel L2 error Temperature: {err_T:.3e}")

# 4) Plot side-by-side
#    render_results, which expects six arguments:
#    V_pred, T_pred, V_exact, T_exact, graph, filename

# For consistency, coerce all to numpy 1D:
V_pred = np.array(V_pred).reshape(-1)
T_pred = np.array(T_pred).reshape(-1)
V_vals = np.array(V_vals_fem).reshape(-1)
T_vals = np.array(T_vals_fem).reshape(-1)

render_results(V_pred, T_pred, V_vals, T_vals, graph, filename="fem_vs_gnn.png")

# ── 3) plot only the PREDICTIONS ─────────────────────────────────────────────

pos_np = graph.pos.cpu().numpy()
x, y   = pos_np[:,0], pos_np[:,1]

fig, axes = plt.subplots(1, 2, figsize=(12,5), tight_layout=True)

# Voltage
sc0 = axes[0].scatter(x, y, c=V_pred.flatten(), cmap='viridis', s=5)
axes[0].set_title("Predicted Voltage")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(sc0, ax=axes[0], shrink=0.7)

# Temperature
sc1 = axes[1].scatter(x, y, c=T_pred.flatten(), cmap='plasma', s=5)
axes[1].set_title("Predicted Temperature")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
plt.colorbar(sc1, ax=axes[1], shrink=0.7)

plt.savefig("coupled_preds.png", dpi=300)
plt.close(fig)
print("Done — predictions plotted to coupled_preds.png")
