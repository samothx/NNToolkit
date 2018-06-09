import numpy as np


class Activation:
    def __init__(self, act_type):
        self.__type = act_type

    def forward(self, z, ** kwargs):
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

    def forward(self, z, ** kwargs):
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
        # anti div by zero

        if "check_overflow" in kwargs:
            if np.min(a) == 0:
                zeros = np.int64(a == 0)
                if np.count_nonzero(zeros) > 0:
                    a = a + (zeros * 1e-8)

            if np.max(a) == 1:
                ones = np.int64(a == 1)
                if np.count_nonzero(ones) > 0:
                    a = a - (ones * 1e-8)

        cost = -np.sum(np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))) / a.shape[1]

        if learn:
            dz = (a - y)
        else:
            dz = None

        return cost, dz


class TanH(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, z, ** kwargs):
        return np.tanh(z)

    def get_grads(self, z, da):
        return np.multiply(1 - np.power(np.tanh(z), 2), da)

    def get_cost(self, a, y, learn=True, **kwargs):
        raise TypeError("get_cost() is not implemented on TanH class")

    def get_yhat(self, a, **kwargs):
        raise TypeError("get_yhat() is not implemented on TanH class")


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")
        self.__ones = None

    def forward(self, z, ** kwargs):
        return np.maximum(0, z)

    def get_grads(self, z, da):
        return np.multiply(np.int64(z > 0), da)

    def get_he_init(self, l_prev):
        return np.sqrt(2 / l_prev)

    def get_cost(self, a, y, learn=True,  **kwargs):
        raise TypeError("get_cost() is not implemented on ReLU class")

    def get_yhat(self, a, **kwargs):
        raise TypeError("get_yhat() is not implemented on ReLU class")


class Softmax(Activation):

    def __init__(self):
        super().__init__("softmax")
        self.__a = None

    def forward(self, z, **kwargs):
        # if "check_overflow" in kwargs:
        #     if kwargs["check_overflow"]:
        #         zeros = np.int64(z == 0)
        #         if np.max(zeros) > 0:
        #             print("add epsilon")
        #             z = z + (zeros * 1e-8)

        t = np.exp(z)
        return t / np.sum(t, axis=0, keepdims=True)
        # raise TypeError("forward() is not implemented on Softmax class")

    def get_grads(self, z, da):
        # TODO: not really needed but sort out anyway..
        raise TypeError("get_grads() is not implemented on Softmax class")

    def get_yhat(self, a, **kwargs):
        # TODO: implement
        y_hat = np.zeros(a.shape)
        y_hat[np.where(a == np.max(a, axis=0))] = 1
        return y_hat

    def get_cost(self, a, y, learn=True, **kwargs):
        # assert isinstance(params, RuntimeParams)
        # TODO: implement

        assert y.shape == a.shape
        # calculate cost function
        # anti div by zero

        if "check_overflow" in kwargs:
            if kwargs["check_overflow"]:
                zeros = np.int64(a == 0)
                if np.count_nonzero(zeros) > 0:
                    a = a + (zeros * 1e-8)

                ones = np.int64(a == 1)
                if np.count_nonzero(ones) > 0:
                    a = a - (ones * 1e-8)

        cost = -np.sum(np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))) / a.shape[1]

        if learn:
            dz = a - y
        else:
            dz = None

        return cost, dz
