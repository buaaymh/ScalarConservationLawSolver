import abc
import numpy as np
import basis_function
import dg_cell
from scipy.integrate import fixed_quad as quad
from scipy.optimize import minimize

class Limiter(abc.ABC):

    @abc.abstractclassmethod
    def limit(self, cell_left, cell_target, cell_right):
        pass

class TVBLimiter(Limiter):
    
    def __init__(self, a_func):
        self._a_func = a_func

    def limit(self, cell_left, cell_target, cell_right):
        is_trouble = self._kxrcf(cell_left, cell_target, cell_right)
        if (is_trouble):
            u_average = cell_target.u_average()
            delta_left = u_average - cell_left.u_average()
            delta_right = cell_right.u_average() - u_average
            mod_left = u_average - cell_target.u_on_left()
            mod_right = cell_target.u_on_right() - u_average
            new_mod_left = self._minmod([mod_left, delta_right, delta_left])
            new_mod_right = self._minmod([mod_right, delta_right, delta_left])
            u_l = u_average - new_mod_left
            u_r = u_average + new_mod_right
            cell_target.reconstruct(u_l, u_r)
    
    def _kxrcf(self, cell_left, cell_target, cell_right):
        u_average = cell_target.u_average()
        a = self._a_func(u_average)
        self._dx = cell_target.get_length()
        p = cell_target.get_coefficients_num() - 1
        if a >= 0:
            q_inlet_neighb = cell_left.u_on_right()
            q_inlet_target = cell_target.u_on_left()
        else:
            q_inlet_neighb = cell_right.u_on_left()
            q_inlet_target = cell_target.u_on_right()
        indicator = np.abs((q_inlet_neighb - q_inlet_target) /
                           ((self._dx) ** ((p+1)/2)) * u_average)
        if indicator > 1:
            return True
        else:
            return False

    def _minmod(self, a):
        a.sort()
        if a[0] * a[2] <= 0:
            return 0.0
        else:
            if a[0] >= 0:
                return a[0]
            else:
                return a[2]

class WENOLimiter(Limiter):

    def __init__(self, a_func):
        self._a_func = a_func
        self._basis = basis_function.ThreeOrderOrthogonalityPolynomial()
    
    def limit(self, cell_left, cell_target, cell_right):
        is_trouble = self._kxrcf(cell_left, cell_target, cell_right)
        if (is_trouble):
            self._weno_reconstruct(cell_left, cell_target, cell_right)

    def _kxrcf(self, cell_left, cell_target, cell_right):
        u_average = cell_target.u_average()
        a = self._a_func(u_average)
        self._dx = cell_target.get_length()
        p = cell_target.get_coefficients_num() - 1
        if a >= 0:
            q_inlet_neighb = cell_left.u_on_right()
            q_inlet_target = cell_target.u_on_left()
        else:
            q_inlet_neighb = cell_right.u_on_left()
            q_inlet_target = cell_target.u_on_right()
        indicator = np.abs((q_inlet_neighb - q_inlet_target) /
                           ((self._dx) ** ((p+1)/2)) * u_average)
        if indicator > 1:
            return True
        else:
            return False

    def _weno_reconstruct(self, cell_left, cell_target, cell_right):
        self._dx = cell_target.get_length()
        coefficients_length = cell_target.get_coefficients_num()
        coefficients_1 = cell_target.get_coefficients()
        head = cell_left._tail
        cell_temp = dg_cell.ThreeOrderDGCell(head, head+self._dx)
        cell_temp.project(lambda x : cell_left.global_polynomial(x))
        coefficients_0 = cell_temp.get_coefficients()
        tail = cell_right._head
        cell_temp = dg_cell.ThreeOrderDGCell(tail-self._dx, tail)
        cell_temp.project(lambda x : cell_right.global_polynomial(x))
        coefficients_2 = cell_temp.get_coefficients()
        smoothness_0 = self._smoothness_indicator(coefficients_0)
        smoothness_1 = self._smoothness_indicator(coefficients_1)
        smoothness_2 = self._smoothness_indicator(coefficients_2)
        omega_0 = 0.001 / (1e-6 + smoothness_0) ** 2
        omega_1 = 0.998 / (1e-6 + smoothness_1) ** 2
        omega_2 = 0.001 / (1e-6 + smoothness_2) ** 2
        # ba = smoothness_0 - smoothness_2
        # omega_0 = 0.001 * (1 + (ba / (smoothness_0 + 1e-6)) ** 2)
        # omega_1 = 0.998 * (1 + (ba / (smoothness_1 + 1e-6)) ** 2)
        # omega_2 = 0.001 * (1 + (ba / (smoothness_2 + 1e-6)) ** 2)
        omega_all = omega_0 + omega_1 + omega_2
        weight_0 = omega_0 / omega_all
        weight_1 = omega_1 / omega_all
        weight_2 = omega_2 / omega_all
        for i in range(1, coefficients_length):
            coefficients_1[i] = (coefficients_0[i] * weight_0 +
                                 coefficients_1[i] * weight_1 +
                                 coefficients_2[i] * weight_2)
        cell_target.set_coefficients(coefficients_1)

    def _smoothness_indicator(self, dof):
        polymomial = lambda x:((dof[0] * self._basis.dv_phi0(x) +
                                dof[1] * self._basis.dv_phi1(x) +
                                dof[2] * self._basis.dv_phi2(x) +
                                dof[3] * self._basis.dv_phi3(x)) ** 2 * 2
                              +(dof[0] * self._basis.dv2_phi0(x) +
                                dof[1] * self._basis.dv2_phi1(x) +
                                dof[2] * self._basis.dv2_phi2(x) +
                                dof[3] * self._basis.dv2_phi3(x)) ** 2 * 8
                              )
        smoothness = quad(polymomial, -1, 1, n=5)[0]
        return smoothness


class HWENOLimiter(Limiter):

    def __init__(self, a_func):
        self._a_func = a_func
        self._basis = basis_function.ThreeOrderOrthogonalityPolynomial()
    
    def limit(self, cell_left, cell_target, cell_right):
        is_trouble = self._kxrcf(cell_left, cell_target, cell_right)
        if (is_trouble):
            self._weno_reconstruct(cell_left, cell_target, cell_right)

    def _kxrcf(self, cell_left, cell_target, cell_right):
        u_average = cell_target.u_average()
        a = self._a_func(u_average)
        self._dx = cell_target.get_length()
        p = cell_target.get_coefficients_num() - 1
        if a >= 0:
            q_inlet_neighb = cell_left.u_on_right()
            q_inlet_target = cell_target.u_on_left()
        else:
            q_inlet_neighb = cell_right.u_on_left()
            q_inlet_target = cell_target.u_on_right()
        indicator = np.abs((q_inlet_neighb - q_inlet_target) /
                           ((self._dx) ** ((p+1)/2)) * u_average)
        if indicator > 1:
            return True
        else:
            return False

    def _weno_reconstruct(self, cell_left, cell_target, cell_right):
        self._dx = cell_target.get_length()
        coefficients_length = cell_target.get_coefficients_num()
        coefficients_1 = cell_target.get_coefficients()
        coefficients_0 = self._get_p0_coefficients(cell_left, coefficients_1[0])
        coefficients_2 = self._get_p2_coefficients(cell_right, coefficients_1[0])
        smoothness_0 = self._smoothness_indicator(coefficients_0)
        smoothness_1 = self._smoothness_indicator(coefficients_1)
        smoothness_2 = self._smoothness_indicator(coefficients_2)
        omega_0 = 0.001 / (1e-6 + smoothness_0) ** 2
        omega_1 = 0.998 / (1e-6 + smoothness_1) ** 2
        omega_2 = 0.001 / (1e-6 + smoothness_2) ** 2
        omega_all = omega_0 + omega_1 + omega_2
        weight_0 = omega_0 / omega_all
        weight_1 = omega_1 / omega_all
        weight_2 = omega_2 / omega_all
        for i in range(1, coefficients_length):
            coefficients_1[i] = (coefficients_0[i] * weight_0 +
                                 coefficients_1[i] * weight_1 +
                                 coefficients_2[i] * weight_2)
        cell_target.set_coefficients(coefficients_1)

    @staticmethod
    def lsf_function(x, coefficients):
        basis = basis_function.ThreeOrderOrthogonalityPolynomial()
        func = lambda a :((coefficients[4] - coefficients[0]) * basis.phi0(a) +
                          (x[0] - coefficients[1]) * basis.phi1(a) +
                          (x[1] - coefficients[2]) * basis.phi2(a) +
                          (x[2] - coefficients[3]) * basis.phi3(a)) ** 2
        return quad(func, -1.0, 1.0, n=7)[0]

    def _get_p0_coefficients(self, cell_left, coefficient_0):
        head = cell_left._head
        tail = cell_left._tail
        coefficients = cell_left.get_coefficients()
        coefficients.append(coefficient_0)
        x = [coefficients[1], coefficients[2], coefficients[3]]
        res = minimize(self.lsf_function, x, args=coefficients, method='SLSQP', tol=1e-6)
        new_coefficients = [coefficient_0, res.x[0], res.x[1], res.x[2]]
        cell_left_copy = dg_cell.ThreeOrderDGCell(head, tail)
        cell_left_copy.set_coefficients(new_coefficients)
        cell_target = dg_cell.ThreeOrderDGCell(tail, tail+self._dx)
        cell_target.project(lambda x : cell_left_copy.global_polynomial(x))
        return cell_target.get_coefficients()

    def _get_p2_coefficients(self, cell_right, coefficient_0):
        head = cell_right._head
        tail = cell_right._tail
        coefficients = cell_right.get_coefficients()
        coefficients.append(coefficient_0)
        x = [coefficients[1], coefficients[2], coefficients[3]]
        res = minimize(self.lsf_function, x, args=coefficients, method='SLSQP', tol=1e-6)
        new_coefficients = [coefficient_0, res.x[0], res.x[1], res.x[2]]
        cell_right_copy = dg_cell.ThreeOrderDGCell(head, tail)
        cell_right_copy.set_coefficients(new_coefficients)
        cell_target = dg_cell.ThreeOrderDGCell(head-self._dx, head)
        cell_target.project(lambda x : cell_right_copy.global_polynomial(x))
        return cell_target.get_coefficients()

    def _smoothness_indicator(self, dof):
        polymomial = lambda x:((dof[0] * self._basis.dv_phi0(x) +
                                dof[1] * self._basis.dv_phi1(x) +
                                dof[2] * self._basis.dv_phi2(x) +
                                dof[3] * self._basis.dv_phi3(x)) ** 2 * 2 +
                               (dof[0] * self._basis.dv2_phi0(x) +
                                dof[1] * self._basis.dv2_phi1(x) +
                                dof[2] * self._basis.dv2_phi2(x) +
                                dof[3] * self._basis.dv2_phi3(x)) ** 2 * 8
                              )
        smoothness = quad(polymomial, -1, 1, n=5)[0]
        return smoothness

if __name__ == '__main__':
    a = [4, 0, 0, 0]
    Limiter = WENOLimiter(a_func=lambda a: 2)
    b = Limiter._smoothness_indicator(a)
    print(b)


