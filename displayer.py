import abc
import numpy as np


class Displayer(abc.ABC):

    @abc.abstractmethod
    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


class ContourDisplayer(Displayer):  

    def __init__(self, x_vec, t_vec, u_mat):
        pass

    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


class AnimationDisplayer(Displayer):

    def __init__(self, x_vec, t_vec, u_mat):
        pass

    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


if __name__ == '__main__':
    c = 1.0
    x_vec = np.linspace(start=0.0, stop=1.0, num=11)
    t_vec = np.linspace(start=0.0, stop=1.0, num=11)
    u_mat = np.zeros((len(t_vec), len(x_vec)))
    u_0 = lambda x: np.sin(x_vec)
    for i in range(len(t_vec)):
        for j in range(len(x_vec)):
            x_0 = x_vec[j] - c * t_vec[i]
            u_mat[i][j] = u_0(x_0)

    d = ContourDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d.display(x_min=0.15, x_max=0.35, t_min=0.15, t_max=0.35)
    d.display()

    d = AnimationDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d.display(x_min=0.0, x_max=1.0, t_min=0.45, t_max=0.55)
    d.display()
