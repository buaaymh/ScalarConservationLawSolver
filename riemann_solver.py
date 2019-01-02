import abc

import numpy as np

from solver import Solver


class RiemannSolver(Solver):

    _eps = 1e-6

    def set_initial(self, u_left, u_right):
        self._u_left = u_left
        self._u_right = u_right
        self._u_mean = (u_left + u_right) / 2
    
    @abc.abstractmethod
    def eval_f_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_matrix_at(self, x_vector, t_vector):
        pass

    def _eval_u_matrix_at(self, x_vector, t_vector):
        assert sorted(x_vector)
        assert sorted(t_vector)
        u_matrix = np.zeros((len(t_vector), len(x_vector)))
        for i in range(len(t_vector)):
            for j in range(len(x_vector)):
                u_matrix[i][j] = self.eval_u_scalar_at(
                    x_scalar=x_vector[j], t_scalar=t_vector[i])
        return u_matrix

class Linear(RiemannSolver):

    def __init__(self, a):
        self._a = a

    def eval_f_scalar_at(self, x_scalar, t_scalar):
        return self._a * self.eval_u_scalar_at(x_scalar, t_scalar)

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        x_0 = x_scalar - self._a * t_scalar
        u = 0
        if x_0 < -self._eps:
            u = self._u_left
        elif x_0 > self._eps:
            u = self._u_right
        else:
            u = self._u_mean
        return u

    def eval_u_matrix_at(self, x_vector, t_vector):
        return self._eval_u_matrix_at(x_vector, t_vector)


class Burgers(RiemannSolver):

    def __init__(self):
        pass

    def _u_shockwave(self, x, t):
        u = 0.0
        slope = x - t * self._u_mean
        if slope > self._eps:
            u = self._u_right
        elif slope < -self._eps:
            u = self._u_left
        else:
            u = self._u_mean
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

    def eval_f_scalar_at(self, x_scalar, t_scalar):
        u = self.eval_u_scalar_at(x_scalar, t_scalar)
        return u**2 / 2

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        if self._u_left >= self._u_right:
            return self._u_shockwave(x_scalar, t_scalar)
        elif self._u_left < self._u_right:
            return self._u_rarefaction(x_scalar, t_scalar)

    def eval_u_matrix_at(self, x_vector, t_vector):
        return self._eval_u_matrix_at(x_vector, t_vector)


if __name__ == '__main__':
    from displayer import ContourDisplayer

    t_vector = np.linspace(start=0.0, stop=1.0, num=11)
    x_vector = np.linspace(start=-1.0, stop=1.0, num=21)
    
    solver = Linear(a=1.0)
    # contact discontinuity
    solver.set_initial(u_left=-1.0, u_right=1.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display()

    solver = Burgers()
    # expansion wave
    solver.set_initial(u_left=-1.0, u_right=1.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display()
    # shock wave
    solver.set_initial(u_left=2.0, u_right=0.0)
    u_matrix = solver.eval_u_matrix_at(x_vector=x_vector, t_vector=t_vector)
    displayer = ContourDisplayer(x_vec=x_vector, t_vec=t_vector, u_mat=u_matrix)
    displayer.display()