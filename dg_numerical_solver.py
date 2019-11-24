import abc
import numpy as np
import dg_cell
import riemann_solver
from dg_limiter import TVBLimiter
from dg_limiter import WENOLimiter
from dg_limiter import HWENOLimiter
import copy

from solver import Solver

class DGSolver(abc.ABC):

    @abc.abstractclassmethod
    def set_mesh(self, cell_num=10, t_num=10, x_num_each_cell=6):
        pass

    def set_u_0(self, u_0_func):
        self._u_0_func = u_0_func
    
    def set_range(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        self._x_min = x_min
        self._x_max = x_max
        self._t_min = t_min
        self._t_max = t_max
    
    def set_flux_func(self, flux_func):
        self._flux_func = flux_func

    def set_riemann_solver(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def set_time_scheme(self, time_scheme):
        self._time_scheme = time_scheme

    def set_limiter(self, limiter):
        self._limiter = limiter

    def _limit(self, cells):
        temp_0 = copy.deepcopy(cells[-2])
        for j in range(len(cells)):
            temp_1 = copy.deepcopy(self._cells[j-1])
            self._limiter.limit(temp_0,cells[j-1],cells[j])
            temp_0 = temp_1

    def _eval_rhs_for_time_scheme(self, cells):
        num = cells[0].get_coefficients_num()
        cells_num = len(cells)
        flux = np.zeros(cells_num)
        for i in range(cells_num):
            u_left = cells[i-1].u_on_right()
            u_right = cells[i].u_on_left()
            flux[i] = self._riemann_solver.eval_flux_on_t_axis(u_left=u_left,
                                                               u_right=u_right)
        dt_of_cofficients = np.zeros((cells_num, num))
        for i in range(cells_num):
            dt_of_cofficients[i-1] = cells[i-1].time_derivate_of_cofficients(
                                                f_left  = flux[i-1],
                                                f_right = flux[i],
                                                f_func  = self._flux_func)
        return dt_of_cofficients / (self._dx / 2)

    def run(self):
        self._x_vec = np.linspace(start=self._x_min, stop=self._x_max-self._dx_mini, num=self._x_num)
        self._t_vec = np.linspace(start=self._t_min, stop=self._t_max, num=self._t_num+1)
        self._u_mat = np.zeros((len(self._t_vec), len(self._x_vec)))
        u_0 = list()
        for i in range(self._cell_num):
            self._cells[i].project(self._u_0_func)
            self._cells[i].set_x_num(self._x_num_each_cell)
            u = self._cells[i].get_u()
            for j in range(self._x_num_each_cell):
                u_0.append(u[j])
        self._limit(self._cells)
        self._u_mat[0] = np.array(u_0)
        # time march
        self._time_scheme.set_rhs_func(
            rhs_func=lambda cells: self._eval_rhs_for_time_scheme(cells))
        for i in range(1, self._t_num+1):
            self._cells = self._time_scheme.get_cells_new(self._cells, self._dt)
            u_new = list()
            for j in range(self._cell_num):
                u = self._cells[j].get_u()
                for k in range(self._x_num_each_cell):
                    u_new.append(u[k])
            self._u_mat[i] = np.array(u_new)

    def get_x_vec(self):
        return self._x_vec
    
    def get_t_vec(self):
        return self._t_vec
    
    def get_u_matrix(self):
        return self._u_mat

class DG3solver(DGSolver):
    def __init__(self):
        pass

    def set_mesh(self, cell_num=10, t_num=10, x_num_each_cell=4):
        self._cell_num = cell_num
        self._t_num = t_num
        self._x_num_each_cell = x_num_each_cell
        self._x_num = cell_num * x_num_each_cell
        self._dx_mini = (self._x_max - self._x_min) / self._x_num
        self._dx = (self._x_max - self._x_min) / self._cell_num
        self._dt = (self._t_max - self._t_min) / self._t_num
        self._cells = list()
        self._x_vec = list()
        head = self._x_min
        for i in range(cell_num):
            self._cells.append(dg_cell.TwoOrderDGCell(head, head+self._dx))
            self._cells[i].set_x_num(num=x_num_each_cell)
            head += self._dx

class DG4solver(DGSolver):
    def __init__(self):
        pass

    def set_mesh(self, cell_num=10, t_num=10, x_num_each_cell=6):
        self._cell_num = cell_num
        self._t_num = t_num
        self._x_num_each_cell = x_num_each_cell
        self._x_num = cell_num * x_num_each_cell
        self._dx_mini = (self._x_max - self._x_min) / self._x_num
        self._dx = (self._x_max - self._x_min) / self._cell_num
        self._dt = (self._t_max - self._t_min) / self._t_num
        self._cells = list()
        self._x_vec = list()
        head = self._x_min
        for i in range(cell_num):
            self._cells.append(dg_cell.ThreeOrderDGCell(head, head+self._dx))
            self._cells[i].set_x_num(num=x_num_each_cell)
            head += self._dx


if __name__ == '__main__':
    def initial(x):
      return np.sign(-x)*3 + 3
      # return x
      # return np.sign(x) * x * 8 - 4
    # DG method
    num_solver = DG4solver()
    # k = 20
    # num_solver.set_u_0(u_0_func=lambda x: np.sin(k * x)/(k * x ))
    # num_solver.set_u_0(u_0_func=lambda x: 3 * np.sin(np.pi * x) + 2)
    num_solver.set_u_0(initial)
    # num_solver.set_riemann_solver(riemann_solver.Linear(a = 1))
    # num_solver.set_flux_func(flux_func=lambda u: 1 * u)
    num_solver.set_riemann_solver(riemann_solver.Burgers())
    num_solver.set_flux_func(flux_func=lambda u: u ** 2 / 2)
    # limiter = TVBLimiter(a_func=lambda a: 1)
    # limiter = WENOLimiter(a_func=lambda a: a)
    limiter = HWENOLimiter(a_func=lambda a: a)
    num_solver.set_limiter(limiter)
    t_min = 0.0
    t_max = 1.0
    t_num = 1000
    num_solver.set_range(x_min=-1.0, x_max=1.0, t_min=t_min, t_max=t_max)
    num_solver.set_mesh(cell_num=40, t_num=t_num, x_num_each_cell=3)
    from time_scheme import ThreeStepDGRungeKutta
    num_solver.set_time_scheme(ThreeStepDGRungeKutta(limiter))
    num_solver.run()
    x_vec = num_solver.get_x_vec()
    t_vec = num_solver.get_t_vec()
    u_mat = num_solver.get_u_matrix()

    from displayer import ContourDisplayer, AnimationDisplayer
    d_animation = AnimationDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d_animation.display(x_min=-1.0, x_max=1.0, t_min=t_min, t_max=t_max)

