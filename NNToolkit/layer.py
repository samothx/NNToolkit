import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import print_matrix


class Layer:
    def __init__(self, params, terminal=False):
        self.__terminal = terminal
        self.__W = None
        self.__b = None
        self.__local_params = False
        self.__size = 0  # number of input nodes to layer
        self.__activation = None  # type of activation (relu, )

        if not terminal:
            assert "size" in params
            self.__size = params["size"]        # number of input nodes to layer

            if "activation" in params:
                self.__activation = params["activation"]
                assert act.Activation in self.__activation.__class__.__bases__

            if "local_params" in params:
                self.__local_params = params["local_params"]
                if "W" in params:
                    self.__W = params["W"]

                if "b" in params:
                    self.__b = params["b"]

        self.__next_layer = None             # the next layer
        self.__layer_idx = 0                # the index of this layer starting with 0
        self.__prev_size = 0                # the size of the previous layer or 0

    def add_layer(self, layer):
        assert (layer.__class__ == Layer) | (Layer in layer.__class__.__bases__)

        if self.__next_layer is not None:
            self.__next_layer.add_layer(layer)
        else:
            self.__next_layer = layer
            layer.__set_params(self.__layer_idx + 1, self.__size)

    def __set_params(self, layer_idx, prev_size):
        if not self.__terminal:
            self.__layer_idx = layer_idx
            self.__prev_size = prev_size

            if self.__layer_idx > 0:
                assert self.__activation is not None

            if self.__local_params:
                if self.__W is None:  # initialize W
                    self.__W = np.random.randn(self.__size, prev_size) * self.__activation.get_he_init(prev_size)
                if self.__b is None:  # initialize b
                    self.__b = np.zeros((self.__size, 1))
        else:
            self.__layer_idx = layer_idx

    def check_ready(self):
        assert self.__size > 0
        assert self.__next_layer is not None
        if self.__local_params & (self.__prev_size > 0):
            assert self.__W.shape == (self.__size, self.__prev_size)
        self.__next_layer.check_ready()

    # this is forward and backward propagation
    # back propagation after recursive call to process
    def process(self, a_in, params):
        if self.__layer_idx == 0:
            # nothing to do in layer 0
            return self.__next_layer.process(a_in, params)

        if self.__local_params:
            # use local parameters
            w = self.__W
            b = self.__b
        else:
            # take parameters from params
            w = params["W" + str(self.__layer_idx)]
            b = params["b" + str(self.__layer_idx)]
            assert w & b & (w.shape == (self.__size, self.__prev_size)) & (b.shape == (self.__size, 1))

        if not (a_in.shape[0] == self.__prev_size):
            print("process() layer:" + str(self.__layer_idx) + " invalid size:" + str(a_in.shape))

        assert a_in.shape[0] == self.__prev_size

        # thats it, forward propagation
        z = np.dot(w, a_in) + b
        a_next = self.__activation.forward(z)

        if "verbose" in params:
            print("w[" + str(self.__layer_idx) + "]: " + print_matrix(w, 6))
            print("b[" + str(self.__layer_idx) + "]: " + print_matrix(b, 6))
            print("z[" + str(self.__layer_idx) + "]: " + print_matrix(z, 6))
            print("a[" + str(self.__layer_idx) + "]: " + print_matrix(a_next, 6))

        # let the next layer do its thing
        res = self.__next_layer.process(a_next, params)

        # do back propagation if requested
        if ("backprop" in params) & (self.__layer_idx > 0):
            da = res["dA"]
            m = da.shape[1]
            dz = self.__activation.get_grads(z, da)

            if "lambda" in params:
                reg = params["lambda"] * np.linalg.norm(w, ord='fro')
                res["cost"] = res["cost"] + reg / (2 * m)
                dw = (np.dot(dz, a_in.T) + reg) / m
            else:
                dw = np.dot(dz, a_in.T) / m

            db = np.sum(dz, axis=1, keepdims=True)/m
            res["dA"] = np.dot(w.T, dz)

            if "verbose" in params:
                print("dz[" + str(self.__layer_idx) + "]:" + print_matrix(dz, 6))
                print("dW[" + str(self.__layer_idx) + "]:" + print_matrix(dw, 6))
                print("db[" + str(self.__layer_idx) + "]:" + print_matrix(db, 6))
                print("dA[" + str(self.__layer_idx - 1) + "]:" + print_matrix(res["dA"], 6))

            if "alpha" in params:
                alpha = params["alpha"]
                w = w - alpha * dw
                b = b - alpha * db

                if self.__local_params:
                    self.__W = w
                    self.__b = b
                else:
                    res["W" + str(self.__layer_idx)] = w
                    res["b" + str(self.__layer_idx)] = b
        return res

    def size(self):
        return self.__size

    def layer_idx(self):
        return self.__layer_idx

    def __str__(self):
        out = " "*self.__layer_idx + "[type: layer idx:" + str(self.__layer_idx) + " size:" + \
              str(self.__size) + " activation:" + str(self.__activation) + "]"
        if self.__next_layer is not None:
            return out + "->\n" + str(self.__next_layer)
        else:
            return out

    def get_weights(self, params):
        if self.__layer_idx > 0:
            assert self.__W is not None
            assert self.__b is not None
            params["W" + str(self.__layer_idx)] = self.__W
            params["b" + str(self.__layer_idx)] = self.__b

        if self.__next_layer is not None:
            self.__next_layer.get_weights(params)
