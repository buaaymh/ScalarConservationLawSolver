import abc


class Displayer(abc.ABC):

    @abc.abstractmethod
    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


class ContourDisplayer(Displayer):  

    def __init__(self, x_vec, t_vec, u_mat):
        pass

    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


class AnimationDisplayer(Displayer):

    def __init__(self, x_vec, t_vec, u_mat):
        pass

    def display(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        pass


if __name__ == '__main__':
    pass
