import abc
from scipy.integrate import fixed_quad as quad
from scipy.optimize import minimize

class TwoOrderOrthogonalityPolynomial():

  def __init__(self):
    self._unitizer_0 = 1 / 2 ** 0.5
    self._unitizer_1 = 3 / 6 ** 0.5
    self._unitizer_2 = 10 ** 0.5 / 2

  def phi0(self, x):
    return self._unitizer_0

  def phi1(self, x):
    return self._unitizer_1 * x

  def phi2(self, x):
    return self._unitizer_2 * 0.5 * (3 * x ** 2 - 1)

  def dv_phi0(self, x):
    return 0.0

  def dv_phi1(self, x):
    return self._unitizer_1

  def dv_phi2(self, x):
    return self._unitizer_2 * 3 * x


class ThreeOrderOrthogonalityPolynomial():

  def __init__(self):
    self._unitizer_0 = 1 / 2 ** 0.5
    self._unitizer_1 = 3 / 6 ** 0.5
    self._unitizer_2 = 10 ** 0.5 / 2
    self._unitizer_3 = 14 ** 0.5 / 2

  def phi0(self, x):
    return self._unitizer_0

  def phi1(self, x):
    return self._unitizer_1 * x

  def phi2(self, x):
    return self._unitizer_2 * 0.5 * (3 * x ** 2 - 1)

  def phi3(self, x):
    return self._unitizer_3 * 0.5 * (5 * x ** 3 - 3 * x)

  def dv_phi0(self, x):
    return 0.0

  def dv_phi1(self, x):
    return self._unitizer_1

  def dv_phi2(self, x):
    return self._unitizer_2 * 3 * x

  def dv_phi3(self, x):
    return self._unitizer_3 * (7.5 * x ** 2 - 1.5)

  def dv2_phi0(self, x):
    return 0.0

  def dv2_phi1(self, x):
    return 0.0

  def dv2_phi2(self, x):
    return self._unitizer_2 * 3

  def dv2_phi3(self, x):
    return self._unitizer_3 * 15 * x

if __name__ == '__main__':
    basis = ThreeOrderOrthogonalityPolynomial()
    # # func = lambda x : basis.phi3(x) * basis.phi3(x)
    # # a = quad(func, -1.0, 1.0, n=5)[0]
    # # print(a)
    # x = [1.0, 2.0]
    # res = minimize(target, x, method='Nelder-Mead', tol=1e-6)
    # print(res.x)