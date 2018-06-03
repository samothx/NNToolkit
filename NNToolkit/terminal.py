import numpy as np
from NNToolkit.layer import Layer
from NNToolkit.util import print_matrix


class Terminal(Layer):
    def __init__(self):
        super().__init__({}, True)

    def add_layer(self, layer):
        raise ValueError("add_layer is not supposed to be called on class Terminal")

    def check_ready(self):
        return True

    def process(self, a, params):
        y_hat = np.zeros(a.shape)
        if a.shape[0] > 1:
            y_hat[np.where(a == np.max(a, axis=0))] = 1
        else:
            y_hat = np.int64(a > 0.5)

        res = {"Y_hat": y_hat}
        # a.argmax(axis=0)
        # y_hat = np.zeros(a.shape)

        if "Y" in params:
            y = params["Y"]
            assert (y is not None) & (y.shape == a.shape)
            # calculate cost function
            # TODO: allow different cost functions / cost function plugin
            res["cost"] = -np.sum(np.multiply(y, np.log(a)) + np.multiply((1-y), np.log(1-a))) / a.shape[1]

            if "backprop" in params:
                res["dA"] = -(np.divide(y, a) - np.divide(1 - y, 1 - a))
                if "verbose" in params:
                    print("Y:    " + print_matrix(y))
                    print("dA[" + str(super().layer_idx() - 1) + "]:" + print_matrix(res["dA"]))

        return res

    def __str__(self):
        return " "*super().layer_idx() + "[type: terminal idx:" + str(super().layer_idx()) + "]"

    def get_weights(self, params):
        pass
