import abc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as colormap
from matplotlib.animation import FuncAnimation


class Displayer(abc.ABC):

    @abc.abstractmethod
    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass

    @staticmethod
    def _find(x_vec, x_min, x_max):
        i_head = 0
        while i_head < len(x_vec) and x_vec[i_head] < x_min:
            i_head += 1
        i_tail = len(x_vec) - 1
        while i_tail > 0 and x_max < x_vec[i_tail]:
            i_tail -= 1
        i_tail += 1
        return i_head, i_tail

    @staticmethod
    def _extract_data(x_vec, x_min, x_max,
                      t_vec, t_min, t_max, u_mat):
        i_head_x, i_tail_x = Displayer._find(x_vec, x_min, x_max)
        i_head_t, i_tail_t = Displayer._find(t_vec, t_min, t_max)
        x_data = x_vec[i_head_x : i_tail_x]
        t_data = t_vec[i_head_t : i_tail_t]
        u_data = u_mat[i_head_t : i_tail_t, i_head_x : i_tail_x]
        return x_data, t_data, u_data


class ContourDisplayer(Displayer):  

    def __init__(self, x_vec, t_vec, u_mat):
        assert len(t_vec) == len(u_mat)
        assert len(x_vec) == len(u_mat[0])
        assert sorted(x_vec)
        assert sorted(t_vec)
        self._x_vec = x_vec
        self._t_vec = t_vec
        self._u_mat = u_mat

    def display(self, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0):
        x_data, t_data, u_data = self._extract_data(
            x_vec=self._x_vec, x_min=x_min, x_max=x_max,
            t_vec=self._t_vec, t_min=t_min, t_max=t_max, u_mat=self._u_mat)
        x_grid, t_grid = np.meshgrid(x_data, t_data)   
        fig, axis = plt.subplots()
        axis.set_aspect('equal')
        cs = axis.contourf(x_grid, t_grid, u_data, cmap=colormap.PuBu_r)
        fig.colorbar(cs)
        plt.show()


class AnimationDisplayer(Displayer):

    def __init__(self, x_vec, t_vec, u_mat):
        assert len(t_vec) == len(u_mat)
        assert len(x_vec) == len(u_mat[0])
        assert sorted(x_vec)
        assert sorted(t_vec)
        self._x_vec = x_vec
        self._t_vec = t_vec
        self._u_mat = u_mat

    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        x_data, t_data, u_data = self._extract_data(
            x_vec=self._x_vec, x_min=x_min, x_max=x_max,
            t_vec=self._t_vec, t_min=t_min, t_max=t_max, u_mat=self._u_mat)
        fig, ax = plt.subplots()
        ln, = ax.plot([], [], 'r.', animated=False)

        def init():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-5.0, 5.0)
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            return ln,

        def update(n):
            ti = "t = {0:.2f}". format(t_data[n])
            ax.set_title(ti)
            ax.figure.canvas.draw()
            ln.set_data(x_data, u_data[n])
            return ln,
        
        animation = FuncAnimation(fig, update, frames=np.arange(0,len(t_data)),
            init_func=init, blit=True, repeat=True, interval=50)
        plt.show()

if __name__ == '__main__':
    c = 1.0
    x_vec = np.linspace(start=0.0, stop=5.0, num=101)
    t_vec = np.linspace(start=0.0, stop=1.0, num=50)
    u_mat = np.zeros((len(t_vec), len(x_vec)))
    u_0 = lambda x: np.sin(x)
    for i in range(len(t_vec)):
        for j in range(len(x_vec)):
            x_0 = x_vec[j] - c * t_vec[i]
            u_mat[i][j] = u_0(x_0)

    d = ContourDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d.display(x_min=0.0, x_max=0.5, t_min=0.15, t_max=0.35)
    d.display()

    d = AnimationDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d.display(x_min=0.0, x_max=5.0, t_min=0.0, t_max=1.0)
    d.display()
