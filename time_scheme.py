import abc

import numpy as np


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

