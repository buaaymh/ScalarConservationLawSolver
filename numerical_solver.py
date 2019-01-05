import abc
import numpy as np

from solver import Solver



class NumericalSolver(Solver):

    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_matrix_at(self, x_vec, t_vec):
        pass


class FluxBasedSolver(NumericalSolver):
    
    def __init__(self):
        pass
    
    def set_u_0(self, u_0_func):
        self._u_0_func = u_0_func

    def set_range(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        self._x_min = x_min
        self._x_max = x_max
        self._t_min = t_min
        self._t_max = t_max

    def set_mesh(self, x_num=10, t_num=10):
        self._x_num = x_num
        self._t_num = t_num
        self._dx = (self._x_max - self._x_min) / self._x_num
        self._dt = (self._t_max - self._t_min) / self._t_num

    def set_numerical_flux(self, numerical_flux):
        self._numerical_flux = numerical_flux

    def set_time_scheme(self, time_scheme):
        self._time_scheme = time_scheme

    def _eval_rhs_for_time_scheme(self, u_vec, dx):
        flux = self._numerical_flux.eval_flux_vec(u_vec=u_vec)
        minus_df = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            minus_df[i] = flux[i-1] - flux[i]
        return minus_df / dx
        
    def run(self):
        x_vec = np.linspace(start=self._dx+self._x_min, stop=self._x_max, num=self._x_num)
        x_vec -= self._dx/2
        t_vec = np.linspace(start=self._t_min, stop=self._t_max, num=1+self._t_num)
        self._u_mat = np.zeros((len(t_vec), len(x_vec)))
        # generate u_0
        u_0 = self._u_0_func
        for j in range(len(x_vec)):
            self._u_mat[0][j] = u_0(x_vec[j])
        # time march
        self._time_scheme.set_rhs_func(
            rhs_func=lambda u_vec: self._eval_rhs_for_time_scheme(u_vec, self._dx))
        for i in range(1, len(t_vec)):
            u_old = self._u_mat[i-1]
            self._u_mat[i] = self._time_scheme.get_u_new(u_old=u_old, dt=self._dt)

    def _find_cell_index(self, x_scalar, t_scalar):
        assert x_scalar >= self._x_min and x_scalar <= self._x_max
        assert t_scalar >= self._t_min and t_scalar <= self._t_max
        x_index = int((x_scalar - self._x_min) / self._dx) % self._x_num
        t_index = int((t_scalar - self._t_min) / self._dt)
        return t_index, x_index
      
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        i, j = self._find_cell_index(x_scalar, t_scalar)
        return self._u_mat[i][j]

    def eval_u_matrix_at(self, x_vec, t_vec):
        u_mat = np.zeros((len(t_vec), len(x_vec)))
        for i in range(len(t_vec)):
            for j in range(len(x_vec)):
                u_mat[i][j] = self.eval_u_scalar_at(x_vec[j], t_vec[i])
        return u_mat


if __name__ == '__main__':
    num_solver = FluxBasedSolver()
    num_solver.set_u_0(u_0_func=lambda x: -0.0-1.0*np.sin(np.pi*x))

    # Choose Numerical Flux
    from numerical_flux import GodunovFlux
    import riemann_solver
    # flux = GodunovFlux(riemann_solver.Linear(a=-1.0))
    flux = GodunovFlux(riemann_solver.Burgers())

    from numerical_flux import RoeFlux
    # flux = RoeFlux(flux_func=lambda x: x**2/2)

    num_solver.set_numerical_flux(flux)
    t_min = 0.0
    t_max = 5.0
    t_num = 200
    num_solver.set_range(x_min=-1.0, x_max=1.0, t_min=t_min, t_max=t_max)
    num_solver.set_mesh(x_num=20, t_num=t_num)

    # Choose time scheme
    from time_scheme import OneStepRungeKutta
    num_solver.set_time_scheme(OneStepRungeKutta())
    num_solver.run()

    # Display results
    x_vec = np.linspace(start=-1.0, stop=1.0, num=101)
    t_vec = np.linspace(start=t_min, stop=t_max, num=1+t_num)
    u_mat = num_solver.eval_u_matrix_at(x_vec=x_vec, t_vec=t_vec)
    from displayer import ContourDisplayer, AnimationDisplayer
    d_animation = AnimationDisplayer(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d_animation.display(x_min=-1.0, x_max=1.0, t_min=t_min, t_max=t_max)
    