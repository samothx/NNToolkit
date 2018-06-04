from NNToolkit.parameters.network import NetworkParams


class ResultParams:

    def __init__(self, y_hat, error=None):
        self.error = error
        self.y_hat = y_hat
        self.cost = 0
        self.da = None
        self.__params = None
        self.__dv = None

    def set_params(self, layer_index, w, b):
        if self.__params is None:
            self.__params = NetworkParams(layer_index, w, b)
        else:
            self.__params.set_params(layer_index, w, b)

    def get_params(self):
        return self.__params

    def has_params(self):
        return self.__params is not None

    def set_derivatives(self, layer_index, dw, db):
        if self.__params is None:
            self.__params = NetworkParams()
            self.__params.set_derivatives(layer_index, dw, db)
        else:
            self.__params.set_derivatives(layer_index, dw, db)
