import torch
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
from core.utils.tools import parse_config, modelTrainer
from functions import CoupledElectroThermalFunc as Func
import matplotlib.pyplot as plt

device = torch.device(0)

out_ndim = 2

ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name  

mesh = ElectrodeMesh(ru=(0.5, 0.5), lb=(-0.5, -0.5), density=65)

graph = mesh.getGraphData().to(device)
graph.pos.requires_grad_()

# 2) Per‐node σ, k
#    (here constant =1, or piecewise define mesh.node_type, etc.)
sigma = torch.ones(graph.num_nodes,1,device=device) * 1.0
kappa = torch.ones(graph.num_nodes,1,device=device) * 1.0

func_main = Func(sigma=sigma, k=kappa, V_D=1.0, T_D=273.0,lb=(-0.5, -0.5), ru=(0.5, 0.5))

model = msgPassing(message_passing_num=3, node_input_size=out_ndim+2, edge_input_size=3, 
                   ndim=out_ndim, device=device, model_dir=ckptpath)    # Mess with MPN# to 2 or 3, +3 comes from source + BC
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
print("mesh")

# Extract node positions and connectivity
pos = mesh.pos  # Shape (N, 2), where N is the number of nodes
faces = mesh.faces  # Shape (3, M), where M is the number of triangular elements

# Plot the mesh
plt.figure(figsize=(8, 8))
plt.triplot(pos[:, 0], pos[:, 1], faces.T, color='blue', linewidth=0.5)
plt.scatter(pos[:, 0], pos[:, 1], color='red', s=1)  
plt.title('Mesh Geometry')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('mesh_plot.png')  # Save the figure to a file
plt.show()

print("bc1 nodes")

on_boundary = torch.squeeze(graph.node_type == ElectrodeMesh.node_type_ref.boundary)  

boundary_indices = torch.where(on_boundary)[0].cpu()

boundary_positions = mesh.pos[boundary_indices.numpy()]

plt.figure(figsize=(8, 8))

# Plot the entire mesh
plt.triplot(mesh.pos[:, 0], mesh.pos[:, 1], mesh.faces.T, color='lightgray')

# Plot the boundary nodes
plt.scatter(boundary_positions[:, 0], boundary_positions[:, 1], color='blue', s=10, label='Boundary Nodes')

plt.title('Boundary and Electrode Nodes')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.savefig('bc1bc2_plot.png')  # Save the figure to a file
plt.show()


    
train_config = parse_config()
writer = SummaryWriter('runs/%s' % Func.func_name)   
 
setattr(train_config, 'pde', func_main.pde_residuals)
setattr(train_config, 'graph_modify', func_main.graph_modify)        
setattr(train_config, 'graph', graph)
setattr(train_config, 'model', model)
setattr(train_config, 'optimizer', optimizer)
setattr(train_config, 'epchoes', 5020)
# **when** to switch domain:
setattr(train_config, 'change_domain_epoch', 5000)
setattr(train_config, 'domain_step', 1000)
# **to what** new domain
setattr(train_config, 'new_lb', (-1.0, -1.0))
setattr(train_config, 'new_ru', ( 1.0,  1.0))

setattr(train_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref) 
setattr(train_config, 'step_times', 1)
setattr(train_config, 'ndim', out_ndim)
setattr(train_config, 'lrstep', 100) #learning rate decay epchoes
setattr(train_config, 'writer', writer)
setattr(train_config, 'func_main', func_main)

modelTrainer(train_config)

