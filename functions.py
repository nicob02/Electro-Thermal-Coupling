
import torch
import torch.autograd as autograd
import numpy as np

class CoupledElectroThermalFunc:

    func_name = 'electro-thermal-coupling'
    def __init__(self, sigma, k, V_D=1.0, T_D=273.0):
        """
        sigma, k: tensors [N,1] giving per‐node conductivities
        V_D(x): Dirichlet voltage on V‐boundary (could be func of x)
        T_D:     Dirichlet temperature on all T‐boundary
        """
        self.sigma = sigma
        self.k     = k
        self.V_D   = V_D
        self.T_D   = T_D

    def graph_modify(self, graph):
        # Build node features once: [x, y, sigma, k]
        pos   = graph.pos                # [N,2]
        graph.x = torch.cat([pos, self.sigma, self.k], dim=-1)
        return graph

    def _ansatz_V(self, graph, Vraw):
        """
        Hard‐enforce V=0 on y=-0.5, V=1 on y=+0.5; Neumann on x=±0.5 is natural.
        G_V(x,y)   = y + 0.5
        D_V(x,y)   = 0.5^2 − y^2
        û_V = G_V + D_V * Vraw
        """
        x = graph.pos[:,0:1]
        y = graph.pos[:,1:2]

        Gv = y + 0.5                   # at y=+0.5 → 1.0, at y=-0.5 → 0.0
        Dv = 0.5**2 - y**2            # zero on the top/bottom, >0 interior

        return Gv + Dv * Vraw


    def _ansatz_T(self, graph, Traw):
        """
        Hard‐enforce T=T_D on all boundaries via
        G_T(x,y)   = T_D
        D_T(x,y)   = (x^2 − 0.5^2)*(y^2 − 0.5^2)
        û_T = G_T + D_T * Traw
        """
        x = graph.pos[:,0:1]
        y = graph.pos[:,1:2]

        Gt = torch.full_like(y, self.T_D)    
        Dt_raw = (x**2 - 0.5**2) * (y**2 - 0.5**2)
        Dt = Dt_raw / (0.5**2 * 0.5**2)    # now max(Dt)=1 at (0,0)
        return Gt + Dt * Traw

    def _gradient(self, u, graph):
        pos = graph.pos
        if not pos.requires_grad:
            pos.requires_grad_()
        return autograd.grad(u, pos,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]   # [N,2]

    def _divergence(self, flux, graph):
        # flux: [N,2] vector at each node
        pos = graph.pos
        if not pos.requires_grad:
            pos.requires_grad_()
        div = torch.zeros(flux.shape[0],1, device=flux.device)
        for i in range(2):
            # differentiate flux[:,i] w.r.t pos[:,i]
            di = autograd.grad(flux[:,i:i+1], pos,
                               grad_outputs=torch.ones_like(flux[:,i:i+1]),
                               create_graph=True)[0][:,i:i+1]
            div = div + di
        return div                          # [N,1]

    def pde_residuals(self, graph, raw):
        """
        raw: [N,2] = [V̂_raw, T̂_raw]
        Returns PV, PT each [N,1].
        """
        # apply hard‐BC ansatz:
        V = self._ansatz_V(graph, raw[:,0:1])
        T = self._ansatz_T(graph, raw[:,1:2])

        # --- Electro PDE ---
        gradV = self._gradient(V, graph)                 # [N,2]
        σ     = self.sigma                               # [N,1]
        fluxV = σ * gradV                                # broadcast→[N,2]
        divV  = self._divergence(fluxV, graph)           # [N,1]
        PV    = divV                                    # should be zero

        # --- Joule heat Q = σ|∇V|²
        Q     = σ * torch.sum(gradV*gradV, dim=1, keepdim=True)

        # --- Thermal PDE ---
        gradT = self._gradient(T, graph)
        k     = self.k
        fluxT = k * gradT
        divT  = self._divergence(fluxT, graph)
        PT    = -divT - Q                                 # = 0
        
        return PV, PT, gradV 
