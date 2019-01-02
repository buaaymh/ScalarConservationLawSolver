import abc
import numpy as np

from solver import Solver



class NumericalSolver(Solver):

    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_matrix_at(self, x_vector, t_vector):
        pass


class FluxBasedSolver(NumericalSolver):
    
    def __init__(self):
        pass
    
    def set_initial(self, func_u_0):
        self._func_u_0 = func_u_0

    def set_range(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        self._x_min = x_min
        self._x_max = x_max
        self._t_min = t_min
        self._t_max = t_max

    def set_mesh(self, x_num=10, t_num=10):
        self._x_num = x_num
        self._t_num = t_num

    def set_numerical_flux(self, numerical_flux):
        self._numerical_flux = numerical_flux

    def set_time_scheme(self, time_scheme):
        self._time_scheme = time_scheme

    def _eval_l_vec(self, u_vec, dx):
        f_vec = self._numerical_flux.eval_f_vec(u_vec)
        l_vec = np.zeros(len(f_vec))
        for i in range(1,len(f_vec)):
            l_vec[i] = f_vec[i] - f_vec[i-1]
        l_vec[0] = f_vec[0] - f_vec[-1]
        return l_vec / dx
        
    def run(self):
        dx = (self._x_max - self._x_min) / self._x_num
        dt = (self._t_max - self._t_min) / self._t_num
        x_vec = np.linspace(start=dx+self._x_min, stop=self._x_max, num=self._x_num)
        x_vec -= dx/2
        t_vec = np.linspace(start=self._t_min, stop=self._t_max, num=1+self._t_num)
        u_mat = np.zeros((len(t_vec), len(x_vec)))
        # generate u_0
        u_0 = self._func_u_0
        for j in range(len(x_vec)):
            u_mat[0][j] = u_0(x_vec[j])
        # time march
        self._time_scheme.set_rhs_func(
            rhs_func=lambda u_vec: self._eval_l_vec(u_vec, dx))
        for i in range(1, len(t_vec)):
            u_old = u_mat[i-1]
            u_mat[i] = self._time_scheme.get_u_new(u_old, dt)

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    def eval_u_matrix_at(self, x_vector, t_vector):
        pass