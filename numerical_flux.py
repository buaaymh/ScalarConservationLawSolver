import abc
import numpy as np


class NumericalFlux(abc.ABC):
    @abc.abstractmethod
    def eval_flux_vec(self, u_vec):
        pass


class GodunovFlux(NumericalFlux):
    
    def __init__(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def eval_flux_vec(self, u_vec):
        flux = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            # u[i-1] ------ f[i-1] ------ u[i]
            # x[i-1] ------ x[i-1/2] ---- x[i]
            u_left = u_vec[i-1]
            u_right = u_vec[i]
            flux[i-1] = self._riemann_solver.eval_flux_on_t_axis(
                              u_left = u_left, u_right = u_right)
        return flux


class RoeFlux(NumericalFlux):
    
    def __init__(self, flux_func):
        self._flux_func = flux_func

    def eval_flux_vec(self, u_vec):
        flux_on_nodes = np.zeros(len(u_vec))
        flux_roe = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
            flux_on_nodes[i] = self._flux_func(u_vec[i])
        for i in range(len(u_vec)):
            u_left, u_right = u_vec[i-1], u_vec[i]
            f_left, f_right = flux_on_nodes[i-1], flux_on_nodes[i]
            if u_left < u_right:
                f_correction = np.abs(f_right - f_left)
            elif u_right < u_left:
                f_correction = -np.abs(f_right - f_left)
            else:
                f_correction = 0.0
            flux_roe[i-1] = (f_left + f_right - f_correction) / 2
        return flux_roe


class Nnd2Flux(NumericalFlux):

    def __init__(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def eval_flux_vec(self, u_vec):
        flux_on_nodes = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
          i -= 1
          norm_left = np.abs(u_vec[i] - u_vec[i-1])
          norm_right = np.abs(u_vec[i+1] - u_vec[i])
          if norm_left < norm_right:
            u_left = (3 * u_vec[i-1] - u_vec[i-2]) / 2
            u_right = (u_vec[i] + u_vec[i-1]) / 2
          else:
            u_left = (u_vec[i-1] + u_vec[i]) / 2
            u_right = (3 * u_vec[i] - u_vec[i+1]) / 2
          flux_on_nodes[i-1] = self._riemann_solver.eval_flux_on_t_axis(
                                     u_left = u_left, u_right = u_right) 
        return flux_on_nodes

class WenoFlux(NumericalFlux):

    def __init__(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def eval_flux_vec(self, u_vec):
        self._u_vec = u_vec
        flux_on_nodes = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
          i -= 2
          u_left = self._eval_positive_u_at(i-1)
          u_right = self._eval_negative_u_at(i)
          flux_on_nodes[i-1] = self._riemann_solver.eval_flux_on_t_axis(
                                     u_left = u_left, u_right = u_right) 
        return flux_on_nodes

    def _eval_positive_u_at(self, i):
        is_1 = (0.250 * (self._u_vec[i-2] -
                     4 * self._u_vec[i-1] +
                     3 * self._u_vec[i]) ** 2 +
                13/12 * (self._u_vec[i-2] -
                     2 * self._u_vec[i-1] +
                         self._u_vec[i]) ** 2)
        is_2 = (0.250 * (self._u_vec[i-1] -
                         self._u_vec[i+1]) ** 2 +
                13/12 * (self._u_vec[i-1] -
                     2 * self._u_vec[i] +
                         self._u_vec[i+1]) ** 2)
        is_3 = (0.250 * (3 * self._u_vec[i] -
                         4 * self._u_vec[i+1] +
                             self._u_vec[i+2]) ** 2 +
                13/12 * (self._u_vec[i] -
                     2 * self._u_vec[i+1] +
                         self._u_vec[i+2]) ** 2)
        ba = is_1 - is_3
        alpha_1 = 0.1 * (1 + (ba / (is_1 + 1e-6)) ** 2)
        alpha_2 = 0.6 * (1 + (ba / (is_2 + 1e-6)) ** 2)
        alpha_3 = 0.3 * (1 + (ba / (is_3 + 1e-6)) ** 2)
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1 = ( 1/3 * self._u_vec[i-2] -
                7/6 * self._u_vec[i-1] +
               11/6 * self._u_vec[i])
        u_2 = (-1/6 * self._u_vec[i-1] +
                5/6 * self._u_vec[i] +
                1/3 * self._u_vec[i+1])
        u_3 = ( 1/3 * self._u_vec[i] +
                5/6 * self._u_vec[i+1] -
                1/6 * self._u_vec[i+2])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

    def _eval_negative_u_at(self, i):
        is_1 = (0.250 * (self._u_vec[i+2] -
                         self._u_vec[i+1] * 4 +
                         self._u_vec[i  ] * 3) ** 2 +
                13/12 * (self._u_vec[i+2] -
                         self._u_vec[i+1] * 2 +
                         self._u_vec[i]) ** 2)
        is_2 = (0.250 * (self._u_vec[i+1] -
                         self._u_vec[i-1]) ** 2 +
                13/12 * (self._u_vec[i+1] -
                         self._u_vec[i] * 2 +
                         self._u_vec[i-1]) ** 2)
        is_3 = (0.250 * (self._u_vec[i] * 3 -
                         self._u_vec[i-1] * 4 +
                         self._u_vec[i-2]) ** 2 +
                13/12 * (self._u_vec[i] -
                         self._u_vec[i-1] * 2 +
                         self._u_vec[i-2]) ** 2)
        ba = is_1 - is_3
        alpha_1 = 0.1 * (1 + (ba / (is_1 + 1e-6)) ** 2)
        alpha_2 = 0.6 * (1 + (ba / (is_2 + 1e-6)) ** 2)
        alpha_3 = 0.3 * (1 + (ba / (is_3 + 1e-6)) ** 2)
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1 = (1/3 * self._u_vec[i+2] -
               7/6 * self._u_vec[i+1] +
              11/6 * self._u_vec[i])
        u_2 = (-1/6 * self._u_vec[i+1] +
                5/6 * self._u_vec[i] +
                1/3 * self._u_vec[i-1])
        u_3 = (1/3 * self._u_vec[i] +
               5/6 * self._u_vec[i-1] -
               1/6 * self._u_vec[i-2])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

class Gvc8Flux(NumericalFlux):

    def __init__(self, riemann_solver):
        self._riemann_solver = riemann_solver

    def eval_flux_vec(self, u_vec):
        flux_on_nodes = np.zeros(len(u_vec))
        for i in range(len(u_vec)):
          i -= 4
          norm_left = np.abs(u_vec[i] - u_vec[i-1])
          norm_right = np.abs(u_vec[i+1] - u_vec[i])
          if norm_left < norm_right:
            u_left = (  17/7000 * u_vec[i-1+5-1] -   283/21000 * u_vec[i-1+5-2] +
                       53/21000 * u_vec[i-1+5-3] +  6269/21000 * u_vec[i-1+5-4] +
                      4429/4200 * u_vec[i-1+5-5] - 10531/21000 * u_vec[i-1+5-6] +
                     4253/21000 * u_vec[i-1+5-7] -    361/7000 * u_vec[i-1+5-8] +
                          3/500 * u_vec[i-1+5-9])
            u_right = (  -4/875 * u_vec[i+5-9] +   893/21000 * u_vec[i+5-8] -
                     4063/21000 * u_vec[i+5-7] + 14510/21000 * u_vec[i+5-6] +
                      2371/4200 * u_vec[i+5-5] -  2299/21000 * u_vec[i+5-4] +
                      137/21000 * u_vec[i+5-3] +     31/7000 * u_vec[i+5-2] -
                         1/1000 * u_vec[i+5-1])
          else:
            u_left =  (  -4/875 * u_vec[i-1+5-1] +   893/21000 * u_vec[i-1+5-2] -
                     4063/21000 * u_vec[i-1+5-3] + 14510/21000 * u_vec[i-1+5-4] +
                      2371/4200 * u_vec[i-1+5-5] -  2299/21000 * u_vec[i-1+5-6] +
                      137/21000 * u_vec[i-1+5-7] +     31/7000 * u_vec[i-1+5-8] -
                         1/1000 * u_vec[i-1+5-9])
            u_right = ( 17/7000 * u_vec[i+5-9] -   283/21000 * u_vec[i+5-8] +
                       53/21000 * u_vec[i+5-7] +  6269/21000 * u_vec[i+5-6] +
                      4429/4200 * u_vec[i+5-5] - 10531/21000 * u_vec[i+5-4] +
                     4253/21000 * u_vec[i+5-3] -    361/7000 * u_vec[i+5-2] +
                          3/500 * u_vec[i+5-1])
          flux_on_nodes[i-1] = self._riemann_solver.eval_flux_on_t_axis(
                                     u_left = u_left, u_right = u_right) 
        return flux_on_nodes