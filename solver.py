import abc


class Solver(abc.ABC):
    
    @abc.abstractmethod
    def eval_u_scalar_at(self, x_scalar, t_scalar):
        pass

    @abc.abstractmethod
    def eval_u_matrix_at(self, x_vector, t_vector):
        pass

if __name__ == '__main__':
    pass
