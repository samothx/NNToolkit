import numpy as np
# import NNToolkit.activation as act
from NNToolkit.parameters.layer import LayerParams
from NNToolkit.parameters.runtime import RuntimeParams
from NNToolkit.parameters.network import NetworkParams
from NNToolkit.parameters.result import ResultParams
from NNToolkit.util import print_matrix

class Layer:

    def __init__(self, params, layer_idx, prev_size):
        assert isinstance(params, LayerParams)
        assert layer_idx > 0

        self.__layer_idx = layer_idx

        self.__prev_size = prev_size
        self.__local_params = params.local_params

        self.__next_layer = None  # the next layer

        self.__size = params.size  # number of input nodes to layer
        self.__activation = params.activation
        if params.weight is not None:
            # pass down precomputed weights
            self.__w = params.weight
            self.__b = params.bias
        else:
            if self.__local_params:
                self.__w = np.random.randn(self.__size, self.__prev_size) * \
                           self.__activation.get_he_init(self.__prev_size)
                self.__b = np.zeros((self.__size, 1))
            else:
                self.__w = None
                self.__b = None

    def add_layer(self, params):
        assert isinstance(params, LayerParams)

        if self.__next_layer is not None:
            self.__next_layer.add_layer(params)
        else:
            self.__next_layer = Layer(params, self.__layer_idx + 1, self.__size)

    def check_ready(self):
        assert self.__size > 0
        assert self.__next_layer is not None
        if self.__local_params & (self.__prev_size > 0):
            assert self.__w.shape == (self.__size, self.__prev_size)
        self.__next_layer.check_ready()

    # this is forward and backward propagation
    # back propagation after recursive call to process
    def process(self, a_prev, params):
        assert isinstance(params, RuntimeParams)

        if self.__local_params:
            # use local parameters
            w = self.__w
            b = self.__b
        else:
            # take parameters from params
            w, b = params.get_params(self.__layer_idx)
            assert (w is not None) & (b is not None)
            assert (w.shape == (self.__size, self.__prev_size)) & (b.shape == (self.__size, 1))

        # if not (a_in.shape[0] == self.__prev_size):
        #     print("process() layer:" + str(self.__layer_idx) + " invalid size:" + str(a_in.shape))

        assert a_prev.shape[0] == self.__prev_size

        # thats it, forward propagation
        # print("layer:" + str(self.__layer_idx) + " w:" + str(w.shape))
        # print("layer:" + str(self.__layer_idx) + " b:" + str(b.shape))
        # print("layer:" + str(self.__layer_idx) + " a:" + str(a_prev.shape))
        z = np.dot(w, a_prev) + b

        a_next = self.__activation.forward(z)

        if params.is_learn() & (self.__next_layer is not None):
            keep_prop = params.get_keep_prob()
            if keep_prop < 1:  # no sense deleting nodesin the output layer
                d = np.int64(np.random.rand(a_next.shape[0], a_next.shape[1]) < keep_prop)
                a_next = np.multiply(d, a_next) / keep_prop
            else:
                d = None
        else:
            d = None
            keep_prop = 1

        if params.is_verbose():
            print("w[" + str(self.__layer_idx) + "]: " + print_matrix(w, 6))
            print("b[" + str(self.__layer_idx) + "]: " + print_matrix(b, 6))
            print("z[" + str(self.__layer_idx) + "]: " + print_matrix(z, 6))
            print("a[" + str(self.__layer_idx) + "]: " + print_matrix(a_next, 6))

        da = None
        dz = None

        if self.__next_layer:
            # let the next layer do its thing
            res, da = self.__next_layer.process(a_next, params)
        else:
            res = ResultParams()

            if params.is_compute_y():
                res.y_hat = self.__activation.get_yhat(a_next, threshold=params.get_threshold())

            if params.is_learn():
                res.cost, dz = self.__activation.get_cost(a_next, params.get_y(), True,
                                                          check_overflow=params.get_check_overflow())

        if res.is_error():
            return res

        m = a_next.shape[1]

        if params.is_learn():
            # print("layer learn:" + str(self.__layer_idx))
            if self.__next_layer is not None:
                if d is not None:
                    da = np.multiply(da, d) / keep_prop

                dz = self.__activation.get_grads(z, da)

            lambd = params.get_lambda()
            if lambd > 0:
                res.cost = res.cost + lambd * np.sum(np.square(w)) / (2 * m)
                dw = (np.dot(dz, a_prev.T) + lambd * w) / m
            else:
                dw = np.dot(dz, a_prev.T) / m

            db = np.sum(dz, axis=1, keepdims=True)/m

            if self.__layer_idx > 1:
                da = np.dot(w.T, dz)

            if params.is_verbose():
                print("dz[" + str(self.__layer_idx) + "]:" + print_matrix(dz, 6))
                print("dW[" + str(self.__layer_idx) + "]:" + print_matrix(dw, 6))
                print("db[" + str(self.__layer_idx) + "]:" + print_matrix(db, 6))
                print("dA[" + str(self.__layer_idx - 1) + "]:" + print_matrix(da, 6))

            if params.is_update():
                w, b = params.update(w, b, dw, db, self.__layer_idx)

                if self.__local_params:
                    self.__w = w
                    self.__b = b
                else:
                    res.set_params(self.__layer_idx, w, b)
            else:
                if not self.__local_params:
                    res.set_derivatives(self.__layer_idx, dw, db)

        else:
            if res.cost is not None:
                lambd = params.get_lambda()
                if lambd > 0:
                    res.cost = res.cost + lambd * np.sum(np.square(w)) / (2 * a_prev.shape[1])

        return res, da

    def set_local_params(self, local):
        self.__local_params = local
        if self.__next_layer:
            self.__next_layer.set_local_params(local)

    def size(self):
        return self.__size

    def layer_idx(self):
        return self.__layer_idx

    def __str__(self):
        out = " "*self.__layer_idx + "[type: layer idx:" + str(self.__layer_idx) + " size:(" + \
              str(self.__prev_size) + " x " + str(self.__size) + ") activation:" + str(self.__activation) + "]"
        if self.__next_layer is not None:
            return out + "->\n" + str(self.__next_layer)
        else:
            return out

    def get_weights(self, params):
        assert isinstance(params, NetworkParams)
        params.set_params(self.__layer_idx, self.__w, self.__b)
        if self.__next_layer is not None:
            self.__next_layer.get_weights(params)
