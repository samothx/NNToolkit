import numpy as np


class Activation:
    def __init__(self, act_type):
        self.__type = act_type

    def forward(self, a):
        raise TypeError("forward() is not meant to called on Activation base class")

    def get_grads(self, z, da):
        raise TypeError("get_grads() is not meant to called on Activation base class")

    def get_cost(self, a, y, learn=True, **kwargs):
        raise TypeError("get_cost() is not meant to called on Activation base class")

    def get_yhat(self, a, **kwargs):
        raise TypeError("get_yhat() is not meant to called on Activation base class")

    def get_he_init(self, l_prev):
        return np.sqrt(1 / l_prev)

    def __str__(self):
        return self.__type


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("sigmoid")
        self.__a = None

    def forward(self, z):
        return np.divide(1, (1 + np.exp(-z)))

    def get_grads(self, z, da):
        tmp = np.exp(z)
        return np.multiply(np.divide(tmp, np.power(tmp + 1, 2)), da)

    def get_yhat(self, a, **kwargs):

        if a.shape[0] > 1:
            # hardmax
            y_hat = np.zeros(a.shape)
            y_hat[np.where(a == np.max(a, axis=0))] = 1
        else:
            if "threshold" in kwargs:
                threshold = kwargs["threshold"]
            else:
                threshold = 0.5
            y_hat = np.int64(a > threshold)

        return y_hat

    def get_cost(self, a, y, learn=True, **kwargs):
        # assert isinstance(params, RuntimeParams)

        assert y.shape == a.shape
        # calculate cost function
        cost = -np.sum(np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))) / a.shape[1]

        if learn:
            da = -(np.divide(y, a) - np.divide(1 - y, 1 - a))
        else:
            da = None

        return cost, da


class TanH(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, z):
        return np.tanh(z)

    def get_grads(self, z, da):
        return np.multiply(1 - np.power(np.tanh(z), 2), da)

    def get_cost(self, a, y, learn=True, **kwargs):
        raise TypeError("get_cost() is implemented on TanH class")

    def get_yhat(self, a, **kwargs):
        raise TypeError("get_yhat() is implemented on TanH class")


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")
        self.__ones = None

    def forward(self, z):
        return np.maximum(0, z)

    def get_grads(self, z, da):
        return np.multiply(np.int64(z > 0), da)

    def get_he_init(self, l_prev):
        return np.sqrt(2 / l_prev)

    def get_cost(self, a, y, learn=True,  **kwargs):
        raise TypeError("get_cost() is implemented on ReLU class")

    def get_yhat(self, a, **kwargs):
        raise TypeError("get_yhat() is implemented on ReLU class")
