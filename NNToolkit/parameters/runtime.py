class RuntimeParams:
    def __init__(self, x=None, y=None, x_cv=None, y_cv=None):
        self.__learn = False
        self.__x = x
        self.__y = y
        self.__x_cv = x_cv
        self.__y_cv = y_cv
        self.__verbose = False
        self.__weight = None
        self.__bias = None
        self.__alpha = 0.01
        self.__lambda = 0
        self.__dump_mode = False
        self.__threshold = 0.5
        self.__max_a = 0

    def valid(self):
        assert self.__x is not None

    # hand down weight instead of weights held in layers
    # makes layers pass back derivatives in results
    def set_params(self, weight, bias):
        self.__weight = weight
        self.__bias = bias

    def get_params(self, layer_idx):
        # weights start at layer 1
        assert layer_idx > 0
        return self.__weight[layer_idx - 1], self.__bias[layer_idx - 1]

    def set_lambda(self, lambd):
        self.__lambda = lambd

    def get_lambda(self):
        return self.__lambda

    def set_alpha(self, alpha):
        self.__alpha = alpha

    def get_alpha(self):
        return self.__alpha

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

    def set_eval(self, x, y=None):
        self.__x = x
        self.__y = y
        self.__learn = False

    def set_train(self, x, y):
        self.__x = x
        self.__y = y
        self.__learn = True

    def is_learn(self):
        return self.__learn

    def get_y(self):
        return self.__y

    def set_cv(self, x, y):
        self.__x_cv = x
        self.__y_cv = y

    def set_verbose(self, verbosity):
        self.__verbose = verbosity

    def is_verbose(self):
        return self.__verbose

    def set_dump_mode(self, on):
        self.__dump_mode = on

    def is_dump_mode(self):
        return self.__dump_mode
