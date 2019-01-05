import abc

import numpy as np

from solver import Solver


class RiemannSolver(Solver):

    _eps = 0e-12

    def set_initial(self, u_left, u_right):
        self._u_left = u_left
        self._u_right = u_right
        self._u_mean = (u_left + u_right) / 2

    def eval_flux_on_t_axis(self, u_left, u_right):
        # flux is always unique on t-axis
        self.set_initial(u_left=u_left, u_right=u_right)
        u_on_t_axis = self.eval_u_scalar_at(x_scalar=0.0, t_scalar=1.0)
        return self._flux(u_on_t_axis)

    @abc.abstractmethod
    def _flux(self, u):
        pass

    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        # return a single u, even on a shock or a contact
        pass

    def eval_u_matrix_at(self, x_vector, t_vector):
        assert sorted(x_vector)
        assert sorted(t_vector)
        u = np.zeros((len(t_vector), len(x_vector)))
        for i in range(len(t_vector)):
            for j in range(len(x_vector)):
                u[i][j] = self.eval_u_scalar_at(
                    x_scalar=x_vector[j], t_scalar=t_vector[i])
        return u


class Linear(RiemannSolver):

    def __init__(self, a):
        self._a = a

    def _flux(self, u):
        return self._a * u

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        # return a single u, even on a contact
        x_0 = x_scalar - self._a * t_scalar
        u = 0.0
        if x_0 > self._eps:
            u = self._u_right
        else:
            u = self._u_left
        return u


class Burgers(RiemannSolver):

    def __init__(self):
        pass

    def _flux(self, u):
        return u**2 / 2

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        if self._u_left > self._u_right:
            return self._u_shockwave(x_scalar, t_scalar)
        elif self._u_left < self._u_right:
            return self._u_rarefaction(x_scalar, t_scalar)
        else:  # self._u_left == self._u_right
            return self._u_right

    def _u_shockwave(self, x, t):
        x_0 = x - t * self._u_mean
        u = 0.0
        if x_0 > self._eps:
            u = self._u_right
        else:
            u = self._u_left
        return u
    
    def _u_rarefaction(self, x, t):
        u = 0.0
        if t == 0.0:
            if x < -self._eps:
                u = self._u_left
            elif x > self._eps:
                u = self._u_right
            else:
                u = self._u_mean
        else:
            slope = x / t
            if  slope < self._u_left:
                u = self._u_left
            elif slope > self._u_right:
                u = self._u_right
            else:
                u = slope
        return u


if __name__ == '__main__':
    from displayer import ContourDisplayer

    t_vector = np.linspace(start=0.0, stop=10.0, num=11)
    x_vector = np.linspace(start=-10.0, stop=10.0, num=21)
    
    solver = Linear(a=-1.0)
    # contact discontinuity
    solver.set_initial(u_left=-1.0, u_right=1.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display(
        x_min=np.min(x_vector), x_max=np.max(x_vector),
        t_min=np.min(t_vector), t_max=np.max(t_vector))

    solver = Burgers()
    # expansion wave
    solver.set_initial(u_left=-1.0, u_right=1.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display(
        x_min=np.min(x_vector), x_max=np.max(x_vector),
        t_min=np.min(t_vector), t_max=np.max(t_vector))
    # shock wave
    solver.set_initial(u_left=0.0, u_right=-2.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display(
        x_min=np.min(x_vector), x_max=np.max(x_vector),
        t_min=np.min(t_vector), t_max=np.max(t_vector))