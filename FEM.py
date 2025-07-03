import torch
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
from functions import CoupledElectroThermalFunc as Func
import matplotlib.pyplot as plt
from fenics import *
import numpy as np
def run_fem(mesh, coords, sigma_vals, k_vals, lb=(-0.5, -0.5), ru=(0.5, 0.5),
            V_D=1.0, T_D=273.0):
    """
    Solve the coupled electro-thermal problem on a rectangular domain [lb, ru].
    Returns:
      coords:     same as input coords (N×2 array)
      V_vals:    pointwise voltage at each input coord
      T_vals:    pointwise temperature at each input coord
    """
    lb_x, lb_y = lb
    ru_x, ru_y = ru

    # 1) extract the underlying FEniCS mesh and build function spaces
    fenics_mesh = mesh.mesh
    V_space = FunctionSpace(fenics_mesh, 'CG', 2)
    T_space = FunctionSpace(fenics_mesh, 'CG', 2)

    # 2) interpolate per-node sigma, kappa onto the DOFs
    dof_coords = V_space.tabulate_dof_coordinates().reshape(-1,2)
    sigma_arr = np.empty((dof_coords.shape[0],), dtype=np.float64)
    k_arr     = np.empty_like(sigma_arr)
    for i, xy in enumerate(dof_coords):
        # pick nearest GNN node
        idx = np.argmin(np.sum((coords - xy)**2, axis=1))
        sigma_arr[i] = sigma_vals[idx]
        k_arr[i]     = k_vals[idx]

    sigma_fun = Function(V_space)
    sigma_fun.vector()[:] = sigma_arr
    k_fun = Function(T_space)
    k_fun.vector()[:] = k_arr

    # 3) boundary markers via lambda‐functions
    tol = 1e-8
    def top(x, on_bnd):
        return on_bnd and near(x[1], ru_y, tol)
    def bottom(x, on_bnd):
        return on_bnd and near(x[1], lb_y, tol)
    def lateral(x, on_bnd):
        return on_bnd and ( near(x[0], lb_x, tol) or near(x[0], ru_x, tol) )

    # 4) Electrostatic solve: ∇·(σ∇V)=0
    V = TrialFunction(V_space)
    v = TestFunction(V_space)
    aV = dot(sigma_fun*grad(V), grad(v))*dx
    LV = Constant(0.0)*v*dx

    bc_top    = DirichletBC(V_space, Constant(V_D), top)
    bc_bottom = DirichletBC(V_space, Constant(0.0), bottom)
    V_sol = Function(V_space)
    solve(aV==LV, V_sol, [bc_top, bc_bottom])

    # 5) Joule heating Q = σ|∇V|²
    V_grad = project(grad(V_sol), VectorFunctionSpace(fenics_mesh,'CG',2))
    Q = project(sigma_fun * dot(V_grad, V_grad), V_space)

    # 6) Thermal solve: ∇·(k∇T) + Q = 0, with T=T_D on all sides
    T = TrialFunction(T_space)
    t = TestFunction(T_space)
    aT = dot(k_fun*grad(T), grad(t))*dx
    LT = Q*t*dx

    bc_T = DirichletBC(T_space, Constant(T_D), 'on_boundary')
    T_sol = Function(T_space)
    solve(aT==LT, T_sol, [bc_T])

    # 7) pointwise evaluation at the original GNN nodes
    N = coords.shape[0]
    V_vals = np.empty((N,), dtype=np.float64)
    T_vals = np.empty((N,), dtype=np.float64)
    for i, (x,y) in enumerate(coords):
        V_vals[i] = V_sol(Point(x,y))
        T_vals[i] = T_sol(Point(x,y))

    return coords, V_vals, T_vals
