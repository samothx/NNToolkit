import numpy as np
import NNLayer.activation as act

verbose = True

class Layer:
    def __init__(self, size, activation,local_params = True,terminal = False):
        if terminal:
            self.__local_params = False
            self.__size = 0                     # number of input nodes to layer
            self.__activation = None            # type of activation (relu, )
        else:
            assert act.Activation in activation.__class__.__bases__

            self.__local_params = local_params
            self.__size = size                  # number of input nodes to layer
            self.__activation = activation      # type of activation (relu, )

        self.__next_layer = None             # the next layer
        self.__layer_idx = 0                # the index of this layer starting with 0
        self.__prev_size = 0                # the size of the previous layer or 0

    def add_layer(self, layer):
        assert (layer.__class__ == Layer) | (Layer in layer.__class__.__bases__)

        if self.__next_layer is not None:
            self.__next_layer.add_layer(layer)
        else:
            self.__next_layer = layer
            layer.__set_params(self.__layer_idx + 1,self.__size)

    def __set_params(self,layer_idx, prev_size):
        self.__layer_idx = layer_idx
        self.__prev_size = prev_size
        if self.__local_params: # initialize w & b
            self.__W = np.random.randn(self.__size, prev_size)
            self.__b = np.zeros((self.__size,1))

    def check_ready(self):
        assert self.__size > 0
        assert self.__next_layer is not None
        if self.__local_params & (self.__prev_size > 0):
            assert self.__W.shape == (self.__size, self.__prev_size)
        self.__next_layer.check_ready()

    def process(self,a_in,params):
        # if verbose:
        #     print ("layer" + str(self.__layer_idx) + ".process()");

        if self.__layer_idx == 0:
            return self.__next_layer.process(a_in,params)

        if self.__local_params:
            w = self.__W
            b = self.__b
        else:
            w = params["W" + str(self.__layer_idx)]
            b = params["b" + str(self.__layer_idx)]
            assert w & b & (w.shape == (self.__size,self.__prev_size)) & (b.shape == (self.__size,1))

        assert a_in.shape[0] == self.__prev_size

        z = np.dot(w,a_in) + b
        a_next = self.__activation.forward(z)

        res = self.__next_layer.process(a_next,params)

        if "backprop" in params:
            da = res["dA"]
            m = da.shape[1]
            dz = self.__activation.get_grads(da);
            dw = np.dot(dz,a_in.T)/m
            db = np.sum(dz,axis=1,keepdims=True)/m
            res["dA"] = np.dot(w.T, dz)
            if "update" in params:
                if self.__local_params:
                    self.__W = w - params["alpha"] * dw
                    self.__b = b - params["alpha"] * db
                else:
                    res["W" + str(self.__layer_idx)] = w - params["alpha"] * dw
                    res["b" + str(self.__layer_idx)] = b - params["alpha"] * db
            else:
                res["dW" + str(self.__layer_idx)] = dw
                res["db" + str(self.__layer_idx)] = db
        return res

    def size(self):
        return self.__size
    def layer_idx(self):
        return self.__layer_idx

    def __str__(self):
        out = " "*self.__layer_idx + "[type: layer idx:" + str(self.__layer_idx) + " size:" + str(self.__size) + " activation:" + str(self.__activation) + "]"
        if self.__next_layer is not None:
            return out + "->\n" + str(self.__next_layer)
        else:
            return out


