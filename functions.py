
import torch
import torch.autograd as autograd
import numpy as np

class CoupledElectroThermalFunc:

    func_name = 'electro-thermal-coupling'
    def __init__(self, sigma, k, V_D=1.0, T_D=273.0, lb=(-0.5, -0.5), ru=(0.5, 0.5)):
        """
        sigma, k: tensors [N,1] giving per‐node conductivities
        V_D(x): Dirichlet voltage on V‐boundary (could be func of x)
        T_D:     Dirichlet temperature on all T‐boundary
        """
        self.sigma = sigma
        self.k     = k
        self.V_D   = V_D
        self.T_D   = T_D
        # store as tensors for broadcasting
        self.lb = torch.tensor(lb, device=sigma.device).view(1,2)
        self.ru = torch.tensor(ru, device=sigma.device).view(1,2)


    def graph_modify(self, graph):
        # Build node features once: [x, y, sigma, k]
        pos   = graph.pos                # [N,2]
        graph.x = torch.cat([pos, self.sigma, self.k], dim=-1)
        return graph

      def _ansatz_V(self, graph, Vraw):
        x = graph.pos[:,0:1]; y = graph.pos[:,1:2]
        # flatten lb/ru into length-2
        lb = self.lb.view(-1)
        ru = self.ru.view(-1)
        lb_x, lb_y = float(lb[0]), float(lb[1])
        ru_x, ru_y = float(ru[0]), float(ru[1])

        # normalize y into [0,1]
        η = (y - lb_y) / (ru_y - lb_y)
        Gv = η                        # → 0 at bottom, 1 at top
        Dv = (ru_y - lb_y)**2 * η*(1-η)  # vanishes at y=lb_y or ru_y
        return Gv + Dv * Vraw

    def _ansatz_T(self, graph, Traw):
        x = graph.pos[:,0:1]; y = graph.pos[:,1:2]
        lb = self.lb.view(-1)
        ru = self.ru.view(-1)
        lb_x, lb_y = float(lb[0]), float(lb[1])
        ru_x, ru_y = float(ru[0]), float(ru[1])

        # shift into [-1,1] for x and y
        ξ = 2*(x - lb_x)/(ru_x - lb_x) - 1
        η = 2*(y - lb_y)/(ru_y - lb_y) - 1
        Gt = torch.full_like(y, self.T_D)
        # vanishes at any of the four edges
        Dt = ((ru_x-lb_x)/2)*((ru_y-lb_y)/2) * (1 - ξ**2)*(1 - η**2)
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
        #gradV = self._gradient(raw[:,0:1], graph)                 # [N,2]
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
