import abc
import numpy as np

from riemann_solver import RiemannSolver

class NumericalFlux(abc.ABC):
    @abc.abstractmethod
    def eval_f_vec(self, u_vec):
        pass


class GodunovFlux(NumericalFlux):
    
    def __init__(self, riemann_solver):
        assert riemann_solver is RiemannSolver
        self._riemann_solver = riemann_solver

    def eval_f_vec(self, u_vec):
        f = np.zeros(len(u_vec))
        for i in range(1,len(u_vec)):
            self._riemann_solver.set_initial(u_left=u_vec[i-1], u_right=u_vec[i+1])
            f[i] = self._riemann_solver.eval_f_scalar_at(x_scalar=0, t_scalar=1.0)
        # periodic condition
        self._riemann_solver.set_initial(u_left=u_vec[-1], u_right=u_vec[0])
        f[0] = self._riemann_solver.eval_f_scalar_at(x_scalar=0, t_scalar=1.0)
        return f


class RoeFlux(NumericalFlux):
    
    def __init__(self):
        pass

    def eval_f_vec(self, u_vec):
        pass

