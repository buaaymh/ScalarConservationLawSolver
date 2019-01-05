import abc
import numpy as np


class Limiter(abc.ABC):

    @abc.abstractclassmethod
    def get_phi_scalar_at(self, r):
        pass

    def get_phi_vector_at(self, r_vec):
        phi_vec = np.zeros(len(r_vec))
        for i in range(len(r_vec)):
            phi_vec[i] = self.get_phi_scalar_at(r_vec[i])
        return phi_vec


class SuperbeeLimiter(Limiter):

    def __init__(self):
        pass

    def get_phi_scalar_at(self, r):
        if r < 0:
            phi = 0
        elif r >= 0 and r < 0.5:
            phi = 2*r
        elif r >= 0.5 and r < 1:
            phi = 1
        elif r >= 1 and r < 2:
            phi = r
        else:
            phi = 2
        return phi
    

class VanleerLimiter(Limiter):

    def __init__(self):
        pass

    def get_phi_scalar_at(self, r):
        if r < 0:
            phi = 0
        else:
            phi = (np.abs(r) + r) / (np.abs(r) + 1)
        return phi


class MinmodLimiter(Limiter):

    def __init__(self):
        pass

    def get_phi_scalar_at(self, r):
        if r < 0:
            phi = 0
        elif r >= 0 and r < 1:
            phi = r
        else:
            phi = 1
        return phi


if __name__ == '__main__':
    r_vec = np.linspace(start=-1.0, stop=3.0, num=1000)
    l1 = SuperbeeLimiter()
    phi1_vec = l1.get_phi_vector_at(r_vec)

    l2 = VanleerLimiter()
    phi2_vec = l2.get_phi_vector_at(r_vec)

    l3 = MinmodLimiter()
    phi3_vec = l3.get_phi_vector_at(r_vec)

    from matplotlib import pyplot as plt
    plt.plot(r_vec, phi1_vec, label='Superbee')
    plt.plot(r_vec, phi2_vec, label='van Leer')
    plt.plot(r_vec, phi3_vec, label='minmod')
    plt.legend()
    plt.show()
