import torch
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
from functions import CoupledElectroThermalFunc as Func
import matplotlib.pyplot as plt
from fenics import *
import numpy as np
import meshio

def run_fem(density=65):
    mesh = ElectrodeMesh(ru=(0.5, 0.5), lb=(-0.5, -0.5), density=density)
    
    V_space = FunctionSpace(mesh, 'CG', 1)
    T_space = FunctionSpace(mesh, 'CG', 1)
    
    # 2) Boundary markers
    tol = 1e-8
    def top_bottom(x, on_bnd):
        return on_bnd and ( near(x[1],0.5,tol) or near(x[1],-0.5,tol) )
    
    def lateral(x, on_bnd):
        return on_bnd and ( near(x[0],0.5,tol) or near(x[0],-0.5,tol) )
    
    # 3) Voltage problem: -Δ V = 0
    V = TrialFunction(V_space)
    v = TestFunction(V_space)
    a_V = dot(grad(V), grad(v))*dx
    L_V = Constant(0)*v*dx
    
    # Dirichlet on top y=0.5 → V=1, bottom y=-0.5 → V=0
    bc_top    = DirichletBC(V_space, Constant(1.0),   lambda x,onb: near(x[1],0.5,tol) and onb)
    bc_bottom = DirichletBC(V_space, Constant(0.0),   lambda x,onb: near(x[1],-0.5,tol) and onb)
    # Neumann on lateral → nothing special (natural)
    bcs_V = [bc_top, bc_bottom]
    
    V_sol = Function(V_space)
    solve(a_V==L_V, V_sol, bcs_V)
    
    # 4) Compute Q = |∇V|^2
    V_grad = project(grad(V_sol), VectorFunctionSpace(mesh,'CG',1))
    Q = project(dot(V_grad, V_grad), V_space)    # scalar Q
    
    # 5) Temperature problem: -Δ T = Q,   T=273 on ALL boundaries
    T = TrialFunction(T_space)
    t = TestFunction(T_space)
    a_T = dot(grad(T), grad(t))*dx
    L_T = Q*t*dx
    
    bc_T = DirichletBC(T_space, Constant(273.0), 'on_boundary')
    T_sol = Function(T_space)
    solve(a_T==L_T, T_sol, [bc_T])
    
    # 6) Extract nodal values for comparison
    coords = V_space.tabulate_dof_coordinates().reshape((-1,2))
    V_vals = V_sol.vector().get_local()
    T_vals = T_sol.vector().get_local()
    
    # 7) Save to file for Paraview / matplotlib
    # convert FEniCS mesh → meshio
    cells = [("triangle", np.array(mesh.cells(),dtype=int))]
    meshio.write_points_cells(
        "fem_coupled.vtk",
        coords,
        cells,
        point_data={"V": V_vals, "T": T_vals}
    )
    print("FEM solve complete. Results in fem_coupled.vtk")
    L_V = Constant(0)*v*dx
    
    # Dirichlet on top y=0.5 → V=1, bottom y=-0.5 → V=0
    bc_top    = DirichletBC(V_space, Constant(1.0),   lambda x,onb: near(x[1],0.5,tol) and onb)
    bc_bottom = DirichletBC(V_space, Constant(0.0),   lambda x,onb: near(x[1],-0.5,tol) and onb)
    
    return coords, V_vals, T_vals
