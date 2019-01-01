import abc

import numpy as np

from solver import Solver


class RiemannSolver(Solver):

    def set_initial(self, u_left, u_right):
        self._u_left = u_left
        self._u_right = u_right
    
    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_matrix_at(self, x_vector, t_vector):
        pass


class Linear(RiemannSolver):

    def __init__(self, a):
        pass

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    def eval_u_matrix_at(self, x_vector, t_vector):
        pass


class Burgers(RiemannSolver):

    def __init__(self):
        pass

    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    def eval_u_matrix_at(self, x_vector, t_vector):
        pass


if __name__ == '__main__':
    from displayer import ContourDisplayer

    t_vector = np.linspace(start=0.0, stop=1.0, num=3)
    x_vector = np.linspace(start=-1.0, stop=1.0, num=5)
    
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