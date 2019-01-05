import abc
import numpy as np


class NumericalFlux(abc.ABC):
    @abc.abstractmethod
    def eval_flux_vec(self, u_vec):
        pass


class GodunovFlux(NumericalFlux):
    
    def __init__(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def eval_flux_vec(self, u_vec):
        flux = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            # u[i-1] ------ f[i-1] ------ u[i]
            # x[i-1] ------ x[i-1/2] ---- x[i]
            u_left = u_vec[i-1]
            u_right = u_vec[i]
            flux[i-1] = self._riemann_solver.eval_flux_on_t_axis(u_left=u_left, u_right=u_right)
        return flux


class RoeFlux(NumericalFlux):
    
    def __init__(self, flux_func):
        self._flux_func = flux_func

    def eval_flux_vec(self, u_vec):
        flux_on_nodes = np.zeros(len(u_vec))
        flux_roe = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            flux_on_nodes[i] = self._flux_func(u_vec[i])
        for i in range(len(u_vec)):
            u_left, u_right = u_vec[i-1], u_vec[i]
            f_left, f_right = flux_on_nodes[i-1], flux_on_nodes[i]
            if u_left < u_right:
                f_correction = np.abs(f_right - f_left)
            elif u_right < u_left:
                f_correction = -np.abs(f_right - f_left)
            else:
                f_correction = 0.0
            flux_roe[i-1] = (f_left + f_right - f_correction) / 2
        return flux_roe


class FirstOrderUpwindFlux(NumericalFlux):

    def __init__(self, flux_func, a_func):
        self._flux_func = flux_func
        self._a_func = a_func

    def eval_flux_vec(self, u_vec):
        a_node = np.zeros(len(u_vec))
        flux_on_nodes = np.zeros(len(u_vec))
        flux_upwind = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            flux_on_nodes[i] = self._flux_func(u_vec[i])
            a_node[i] = self._a_func(u_vec[i])
        for i in range(len(u_vec)):
            i -= 1
            if a_node[i] > 0 and a_node[i] > 0:
                flux_upwind[i] = flux_on_nodes[i]
            elif a_node[i] < 0 and a_node[i] < 0:
                flux_upwind[i] = flux_on_nodes[i+1]
            else:
                flux_upwind[i] = 0
        return flux_upwind