import abc
import numpy as np
import basis_function
from scipy.integrate import fixed_quad as quad



class DGCell(abc.ABC):
    @abc.abstractmethod
    def u_on_left(self):
      pass

    @abc.abstractmethod
    def u_on_right(self):
      pass


class TwoOrderDGCell(DGCell):

    def __init__(self, head, tail):
        self._head = head
        self._tail = tail
        self._basis = basis_function.TwoOrderOrthogonalityPolynomial()
        self._center_x = (self._head + self._tail) / 2
        self._length = self._tail - self._head

    def _local_to_global(self, x_local):
        x_global = self._center_x + x_local * self._length / 2
        return x_global

    def project(self, func):
        f0 = lambda x : self._basis.phi0(x) * func(self._local_to_global(x))
        self._coefficient_0 = quad(f0, -1.0, 1.0, n=5)[0]
        f1 = lambda x : self._basis.phi1(x) * func(self._local_to_global(x))
        self._coefficient_1 = quad(f1, -1.0, 1.0, n=5)[0]
        f2 = lambda x : self._basis.phi2(x) * func(self._local_to_global(x))
        self._coefficient_2 = quad(f2, -1.0, 1.0, n=5)[0]

    def set_x_num(self, num=4):
        step = 2.0 / num
        self._x_display_points = np.arange(-1, 1, step)

    @staticmethod
    def get_coefficients_num():
        return 3

    def get_u(self):
        u = np.zeros(len(self._x_display_points))
        for i in range(len(self._x_display_points)):
            u[i] = self._polynomial(self._x_display_points[i])
        return u

    def set_coefficients(self, coefficient_new):
        self._coefficient_0 = coefficient_new[0]
        self._coefficient_1 = coefficient_new[1]
        self._coefficient_2 = coefficient_new[2]
    
    def get_coefficients(self):
        return [self._coefficient_0, self._coefficient_1, self._coefficient_2]

    def _polynomial(self, x_local):
        return (self._coefficient_0 * self._basis.phi0(x_local) +
                self._coefficient_1 * self._basis.phi1(x_local) +
                self._coefficient_2 * self._basis.phi2(x_local))
    
    def u_average(self):
        return self._coefficient_0 / 2 ** 0.5

    def u_on_left(self):
        return self._polynomial(-1.0)
    
    def u_on_right(self):
        return self._polynomial(+1.0)

    def time_derivate_of_cofficients(self, f_left, f_right, f_func):
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi0(x)
        dt_cofficient_0 = (quad(f, -1.0, 1.0, n=5)[0] +
                          f_left * self._basis.phi0(-1.0) -
                          f_right * self._basis.phi0(1.0))
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi1(x)
        dt_cofficient_1 = (quad(f, -1.0, 1.0, n=5)[0] +
                          f_left * self._basis.phi1(-1.0) -
                          f_right * self._basis.phi1(1.0))
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi2(x)
        dt_cofficient_2 = (quad(f, -1.0, 1.0, n=5)[0] +
                          f_left * self._basis.phi2(-1.0) -
                          f_right * self._basis.phi2(1.0))
        return [dt_cofficient_0, dt_cofficient_1, dt_cofficient_2]
    
    def reconstruct(self, u_l, u_r):
        A = [[self._basis.phi1(-1), self._basis.phi2(-1)],
             [self._basis.phi1(+1), self._basis.phi2(+1)]]
        s = [u_l - self._coefficient_0 / 2 ** 0.5,
             u_r - self._coefficient_0 / 2 ** 0.5]
        r =  np.linalg.solve(A, s)
        self._coefficient_1 = r[0]
        self._coefficient_2 = r[1]

    def get_length(self):
        return self._length
        
class ThreeOrderDGCell(DGCell):

    def __init__(self, head, tail):
        self._head = head
        self._tail = tail
        self._basis = basis_function.ThreeOrderOrthogonalityPolynomial()
        self._center_x = (self._head + self._tail) / 2
        self._length = self._tail - self._head

    def _local_to_global(self, x_local):
        x_global = self._center_x + x_local * self._length / 2
        return x_global

    def _global_to_local(self, x_global):
        x_local = (x_global - self._center_x) / self._length * 2
        return x_local

    def project(self, func):
        f0 = lambda x : self._basis.phi0(x) * func(self._local_to_global(x))
        self._coefficient_0 = quad(f0, -1.0, 1.0, n=6)[0]
        f1 = lambda x : self._basis.phi1(x) * func(self._local_to_global(x))
        self._coefficient_1 = quad(f1, -1.0, 1.0, n=6)[0]
        f2 = lambda x : self._basis.phi2(x) * func(self._local_to_global(x))
        self._coefficient_2 = quad(f2, -1.0, 1.0, n=6)[0]
        f3 = lambda x : self._basis.phi3(x) * func(self._local_to_global(x))
        self._coefficient_3 = quad(f3, -1.0, 1.0, n=6)[0]

    def set_x_num(self, num=6):
        step = 2.0 / num
        self._x_display_points = np.arange(-1, 1, step)
    
    @staticmethod
    def get_coefficients_num():
        return 4

    def get_u(self):
        u = np.zeros(len(self._x_display_points))
        for i in range(len(self._x_display_points)):
            u[i] = self._polynomial(self._x_display_points[i])
        return u

    def set_coefficients(self, coefficient_new):
        self._coefficient_0 = coefficient_new[0]
        self._coefficient_1 = coefficient_new[1]
        self._coefficient_2 = coefficient_new[2]
        self._coefficient_3 = coefficient_new[3]
    
    def get_coefficients(self):
        return [self._coefficient_0, self._coefficient_1,
                self._coefficient_2, self._coefficient_3]

    def _polynomial(self, x_local):
        return (self._coefficient_0 * self._basis.phi0(x_local) +
                self._coefficient_1 * self._basis.phi1(x_local) +
                self._coefficient_2 * self._basis.phi2(x_local) +
                self._coefficient_3 * self._basis.phi3(x_local))
    
    def global_polynomial(self, x_global):
        x_local = self._global_to_local(x_global)
        return self._polynomial(x_local)

    def u_on_left(self):
        return self._polynomial(-1.0)
    
    def u_on_right(self):
        return self._polynomial(+1.0)

    def u_average(self):
        return self._coefficient_0 / 2 ** 0.5

    def time_derivate_of_cofficients(self, f_left, f_right, f_func):
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi0(x)
        dt_cofficient_0 = (quad(f, -1.0, 1.0, n=7)[0] +
                          f_left * self._basis.phi0(-1.0) -
                          f_right * self._basis.phi0(1.0))
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi1(x)
        dt_cofficient_1 = (quad(f, -1.0, 1.0, n=7)[0] +
                          f_left * self._basis.phi1(-1.0) -
                          f_right * self._basis.phi1(1.0))
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi2(x)
        dt_cofficient_2 = (quad(f, -1.0, 1.0, n=7)[0] +
                          f_left * self._basis.phi2(-1.0) -
                          f_right * self._basis.phi2(1.0))
        f = lambda x : f_func(self._polynomial(x)) * self._basis.dv_phi3(x)
        dt_cofficient_3 = (quad(f, -1.0, 1.0, n=7)[0] +
                          f_left * self._basis.phi3(-1.0) -
                          f_right * self._basis.phi3(1.0))
        return [dt_cofficient_0, dt_cofficient_1,
                dt_cofficient_2, dt_cofficient_3]
        
    def reconstruct(self, u_l, u_r):
        A = [[self._basis.phi1(-1), self._basis.phi2(-1)],
             [self._basis.phi1(+1), self._basis.phi2(+1)]]
        s = [u_l - self._coefficient_0 / 2 ** 0.5,
             u_r - self._coefficient_0 / 2 ** 0.5]
        r =  np.linalg.solve(A, s)
        self._coefficient_1 = r[0]
        self._coefficient_2 = r[1]
        self._coefficient_3 = 0

    def get_length(self):
        return self._length

if __name__ == '__main__':
    def func(x):
        return x ** 2

    

    