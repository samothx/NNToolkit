from NNToolkit.parameters.network import NetworkParams
from NNToolkit.update import Update, NoUpdate


class RuntimeParams:
    def __init__(self):
        self.__learn = False
        self.__y = None
        self.__verbose = False
        self.__params = NetworkParams()
        self.__weight = None
        self.__bias = None
        self.__alpha = 0.01
        self.__update = NoUpdate()
        self.__lambda = 0
        self.__threshold = 0.5
        self.__max_a = 0

    # hand down weight instead of weights held in layers
    # makes layers pass back derivatives in results
    def set_params(self, params):
        self.__params = params

    def get_params(self, layer_idx):
        return self.__params.get_params(layer_idx)

    def inc(self,alpha = 0):
        self.__update.next_it(alpha)

    def set_update(self,update):
        assert isinstance(update,Update)
        self.__update = update

    def is_update(self):
        return self.__update.is_update

    def update(self,w , b , vw, vb, layer_idx):
        return self.__update.update(w , b , vw, vb, layer_idx)

    def set_lambda(self, lambd):
        self.__lambda = lambd

    def get_lambda(self):
        return self.__lambda

    def set_max_a(self, max_a):
        assert max_a >= 0
        self.__max_a = max_a

    def get_max_a(self):
        return self.__max_a

    def set_threshold(self, threshold):
        assert (threshold < 1) & (threshold > 0)
        self.__threshold = threshold

    def get_threshold(self):
        return self.__threshold

    def set_eval(self, y=None):
        self.__y = y
        self.__learn = False

    def set_train(self, y):
        self.__y = y
        self.__learn = True

    def is_learn(self):
        return self.__learn

    def get_y(self):
        return self.__y

    # def set_cv(self, x, y):
    #     self.__x_cv = x
    #     self.__y_cv = y

    def set_verbose(self, verbosity):
        self.__verbose = verbosity

    def is_verbose(self):
        return self.__verbose

 #   def set_dump_mode(self, on):
 #       self.__dump_mode = on

 #   def is_dump_mode(self):
 #       return self.__dump_mode
