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
        self.__keep_prob = 1
        self.__threshold = 0.5
        # self.__max_z = 0
        self.__compute_y = False
        self.__check_overflow = False

    # hand down weight instead of weights held in layers
    # makes layers pass back derivatives in results
    def set_params(self, params):
        self.__params = params

    def get_params(self, layer_idx):
        return self.__params.get_params(layer_idx)

    def inc(self, alpha=0):
        self.__update.next_it(alpha)

    def set_update(self, update):
        assert isinstance(update, Update)
        self.__update = update

    def is_update(self):
        return self.__update.is_update

    def update(self, w, b, vw, vb, layer_idx):
        return self.__update.update(w, b, vw, vb, layer_idx)

    def set_lambda(self, lambd):
        self.__lambda = lambd

    def get_lambda(self):
        return self.__lambda

    def set_keep_prob(self, prob):
        assert (prob > 0) & (prob <= 1)
        self.__keep_prob = prob

    def get_keep_prob(self):
        return self.__keep_prob

    def set_chck_overflow(self, co):
        self.__check_overflow = co

    def get_check_overflow(self):
        return self.__check_overflow

    # def set_max_z(self, max_z):
    #     assert max_z >= 0
    #     self.__max_z = max_z

    # def get_max_z(self):
    #     return self.__max_z

    def set_threshold(self, threshold):
        assert (threshold < 1) & (threshold > 0)
        self.__threshold = threshold

    def get_threshold(self):
        return self.__threshold

    def set_compute_y(self, compute_y):
        self.__compute_y = compute_y

    def set_eval(self, y=None):
        self.__y = y
        self.__learn = False
        self.__compute_y = True

    def set_train(self, y, compute_y=False):
        self.__y = y
        self.__learn = True
        self.__compute_y = compute_y

    def is_learn(self):
        return self.__learn

    def is_compute_y(self):
        return self.__compute_y

    def get_y(self):
        return self.__y

    def set_verbose(self, verbosity):
        self.__verbose = verbosity

    def is_verbose(self):
        return self.__verbose
