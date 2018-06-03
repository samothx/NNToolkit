import numpy as np


class Activation:
    def __init__(self, act_type):
        self.__type = act_type

    def forward(self, a):
        assert self.__type is not None

    def get_grads(self, z, da):
        assert self.__type is not None

    def get_he_init(self, l_prev):
        return np.sqrt(1 / l_prev)

    def __str__(self):
        return self.__type


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, z):
        return np.divide(1, (1 + np.exp(-z)))

    def get_grads(self, z, da):
        tmp = np.exp(z)
        return np.multiply(np.divide(tmp, np.power(tmp + 1, 2)), da)
        # a = np.divide(1,1+np.exp(-z))
        # return np.multiply(a,(a + 1))


class TanH(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, z):
        return np.tanh(z)

    def get_grads(self, z, da):
        return np.multiply(1 - np.power(np.tanh(z), 2), da)


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")
        self.__ones = None

    def forward(self, z):
        return np.maximum(0, z)

    def get_grads(self, z, da):
        if self.__ones is None:
            self.__ones = np.ones(z.shape)
        elif z.shape != self.__ones.shape:
            self.__ones = np.ones(z.shape)

        return np.multiply(self.__ones * (z > 0), da)

    def get_he_init(self,l_prev):
        return np.sqrt(2 / l_prev)
