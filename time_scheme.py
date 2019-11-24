import abc
from dg_limiter import WENOLimiter
import numpy as np
import copy



class TimeScheme(abc.ABC):

    @abc.abstractclassmethod
    def get_u_new(self, u_old, dt):
        pass

    def set_rhs_func(self, rhs_func):
        self._rhs = rhs_func


class OneStepRungeKutta(TimeScheme):
    
    def __init__(self):
        pass

    def get_u_new(self, u_old, dt):
        return u_old + dt * self._rhs(u_old)

class TwoStepRungeKutta(TimeScheme):
    
    def __init__(self):
        pass

    def get_u_new(self, u_old, dt):
        u_1 = u_old + dt * self._rhs(u_old)
        u_new = 0.5 * u_old + 0.5 * u_1 + 0.5 * dt * self._rhs(u_1)
        return u_new

class ThreeStepRungeKutta(TimeScheme):
    
    def __init__(self):
        pass

    def get_u_new(self, u_old, dt):
        u_1 = u_old + dt * self._rhs(u_old)
        u_2 = (3 * u_old + u_1 + dt * self._rhs(u_1))/4
        u_new = (u_old + 2*(u_2 + dt * self._rhs(u_2)))/3
        return u_new

class ThreeStepDGRungeKutta():
    
    def __init__(self, limiter):
        self._limiter = limiter
    
    def set_rhs_func(self, rhs_func):
        self._rhs = rhs_func

    def get_cells_new(self, cells_old, dt):
        self._cells = cells_old
        # step 1
        coefficient_old = self._get_coefficients(self._cells)
        coefficient_1 = coefficient_old + dt * self._rhs(self._cells)
        self._update(self._cells, coefficient_1)
        self._limit(self._cells)
        # step 2
        coefficient_1 = self._get_coefficients(self._cells)
        coefficient_2 = (3 * coefficient_old + coefficient_1 + dt * self._rhs(self._cells))/4
        self._update(self._cells, coefficient_2)
        self._limit(self._cells)
        # step 3
        coefficient_2 = self._get_coefficients(self._cells)
        coefficient_new = (coefficient_old + 2*(coefficient_2 + dt * self._rhs(self._cells)))/3
        self._update(self._cells, coefficient_new)
        self._limit(self._cells)
        return self._cells
    
    def _limit(self, cells):
        temp_0 = copy.deepcopy(cells[-2])
        for j in range(len(cells)):
            temp_1 = copy.deepcopy(self._cells[j-1])
            self._limiter.limit(temp_0,cells[j-1],cells[j])
            temp_0 = temp_1
    
    def _update(self, cells, coefficients):
        for i in range(len(cells)):
            cells[i].set_coefficients(coefficients[i])
    
    def _get_coefficients(self, cells):
        num = cells[0].get_coefficients_num()
        coefficients = np.zeros((len(cells), num))
        for i in range(len(cells)):
              coefficients[i] = self._cells[i].get_coefficients()
        return coefficients


if __name__ == '__main__':
    dt = 0.1
    u_0_vec = np.array([1.0, 0.0])

    def rhs(u_vec):
        return np.exp(-u_vec)

    t_marcher = OneStepRungeKutta()
    t_marcher.set_rhs_func(rhs_func=rhs)
    u_new = t_marcher.get_u_new(u_old=u_0_vec, dt=dt)
    print(u_new)
    
    t_marcher = ThreeStepRungeKutta()
    t_marcher.set_rhs_func(rhs_func=rhs)
    u_new = t_marcher.get_u_new(u_old=u_0_vec, dt=dt)
    print(u_new)

