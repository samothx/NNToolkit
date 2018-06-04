import numpy as np
# import NNToolkit.activation as act
from NNToolkit.parameters.layer import LayerParams
from NNToolkit.parameters.runtime import RuntimeParams
from NNToolkit.parameters.network import NetworkParams
from NNToolkit.parameters.result import ResultParams
from NNToolkit.util import print_matrix


class Layer:

    def __init__(self, params, layer_idx, prev_size, terminal=False):
        assert isinstance(params, LayerParams)
        assert layer_idx > 0

        self.__terminal = terminal
        self.__layer_idx = layer_idx

        self.__prev_size = prev_size
        self.__local_params = params.local_params

        self.__next_layer = None  # the next layer

        if terminal:
            self.__w = None
            self.__b = None
            self.__activation = None
            self.__size = 0
        else:
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

    def add_layer(self, params, terminal=False):
        assert isinstance(params, LayerParams)
        assert self.__terminal is False

        if self.__next_layer is not None:
            self.__next_layer.add_layer(params, terminal)
        else:
            if terminal:
                self.__next_layer = Terminal(params, self.__layer_idx + 1, self.__size)
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
            assert w & b
            assert (w.shape == (self.__size, self.__prev_size)) & (b.shape == (self.__size, 1))

        # if not (a_in.shape[0] == self.__prev_size):
        #     print("process() layer:" + str(self.__layer_idx) + " invalid size:" + str(a_in.shape))

        assert a_prev.shape[0] == self.__prev_size

        # thats it, forward propagation
        z = np.dot(w, a_prev) + b
        a_next = self.__activation.forward(z)

        if params.is_verbose():
            print("w[" + str(self.__layer_idx) + "]: " + print_matrix(w, 6))
            print("b[" + str(self.__layer_idx) + "]: " + print_matrix(b, 6))
            print("z[" + str(self.__layer_idx) + "]: " + print_matrix(z, 6))
            print("a[" + str(self.__layer_idx) + "]: " + print_matrix(a_next, 6))

        # let the next layer do its thing
        res = self.__next_layer.process(a_next, params)

        # do back propagation if requested
        if params.is_learn():
            da = res.da
            m = da.shape[1]
            dz = self.__activation.get_grads(z, da)

            lambd = params.get_lambda()
            if lambd > 0:
                res.cost = res.cost + lambd * np.sum(np.square(w)) / (2 * m)
                dw = (np.dot(dz, a_prev.T) + lambd * w) / m
            else:
                dw = np.dot(dz, a_prev.T) / m

            db = np.sum(dz, axis=1, keepdims=True)/m

            if self.__layer_idx > 1:
                res.da = np.dot(w.T, dz)

            if params.is_verbose():
                print("dz[" + str(self.__layer_idx) + "]:" + print_matrix(dz, 6))
                print("dW[" + str(self.__layer_idx) + "]:" + print_matrix(dw, 6))
                print("db[" + str(self.__layer_idx) + "]:" + print_matrix(db, 6))
                print("dA[" + str(self.__layer_idx - 1) + "]:" + print_matrix(res["dA"], 6))

            alpha = params.get_alpha()
            if alpha > 0:
                w = w - alpha * dw
                b = b - alpha * db

            if self.__local_params:
                self.__w = w
                self.__b = b
            else:
                res.set_derivatives(self.__layer_idx, dw, db)

            if params.is_dump_mode():
                res.set_params(self.__layer_idx, w, b)
                if self.__local_params:
                    res.set_derivatives(self.__layer_idx, dw, db)

        return res

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
        if not self.__terminal:
            assert isinstance(params, NetworkParams)
            params.set_params(self.__layer_idx, self.__w, self.__b)
            if self.__next_layer is not None:
                self.__next_layer.get_weights(params)


class Terminal(Layer):
    def __init__(self, params, layer_idx, prev_size):
        assert isinstance(params, LayerParams)
        super().__init__(params, layer_idx, prev_size, True)

    def check_ready(self):
        return True

    def process(self, a_prev, params):
        assert isinstance(params, RuntimeParams)

        max_a = params.get_max_a()
        if max_a & (np.linalg.norm(a_prev) > max_a):
            size = np.linalg.norm(a_prev)
            if size > max_a:
                return ResultParams(None, "a exceeded max size:" + str(size) + ">" + str(max_a))

        m = a_prev.shape[1]
        n = a_prev.shape[0]

        if n > 1:
            y_hat = np.zeros((n, m))
            y_hat[np.where(a_prev == np.max(a_prev, axis=0))] = 1
        else:
            y_hat = np.int64(a_prev > params.get_threshold())

        res = ResultParams(y_hat)

        y = params.get_y()
        if y is not None:
            assert (y.shape == a_prev.shape)
            # calculate cost function
            # TODO: allow different cost functions / cost function plugin
            res.cost = -np.sum(np.multiply(y, np.log(a_prev)) + np.multiply((1-y), np.log(1-a_prev))) / m

            if params.is_learn():
                res.da = -(np.divide(y, a_prev) - np.divide(1 - y, 1 - a_prev))
                if params.is_verbose():
                    print("Y:    " + print_matrix(y))
                    print("dA[" + str(super().layer_idx() - 1) + "]:" + print_matrix(res.da))

        return res

    def __str__(self):
        return " "*super().layer_idx() + "[type: terminal idx:" + str(super().layer_idx()) + "]"
