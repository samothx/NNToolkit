import numpy as np


class Activation:
    def __init__(self,type):
        self.__type = type

    def forward(self, a):
        assert self.__type is not None

    def get_grads(self):
        assert self.__type is not None

    def __str__(self):
        return self.__type


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, a):
        return np.divide(1,(1 + np.exp(-a)))

    def get_grads(self,a):
        a_tmp = np.exp(-a)
        return np.divide(a_tmp,np.power(a_tmp + 1, 2))


class TanH(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, a):
        return np.tanh(a)

    def get_grads(self,a):
        a_tmp = np.exp(-a)
        return 1 - np.power(np.tanh(a),2)


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")
        self.__ones = None

    def forward(self, a):
        return a * (a > 0)

    def get_grads(self,a):
        if (self.__ones is None):
            self.__ones = np.ones(a.shape)
        elif (a.shape != self.__ones.shape):
            self.__ones = np.ones(a.shape)

        return self.__ones * (a > 0)