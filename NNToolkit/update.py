import numpy as np


class Update:
    def __init__(self, upd_type, is_upd):
        self.__type = upd_type
        self.is_update = is_upd

    def next_it(self, alpha):
        raise TypeError("reset() should not be called on base class")

    def reset(self):
        raise TypeError("reset() should not be called on base class")

    def update(self, w, b, dw, db, layer_idx):
        raise TypeError("update() should not be called on base class")


class NoUpdate(Update):

    def __init__(self):
        super().__init__("none", False)

    def next_it(self):
        pass

    def reset(self, alpha):
        pass

    def update(self, w, b, dw, db, layer_idx):
        return w, b


class StandardUpd(Update):

    def __init__(self, alpha):
        super().__init__("standard", True)
        self.__alpha = alpha

    def next_it(self, alpha = 0):
        if alpha:
            self.__alpha = alpha

    def reset(self):
        pass

    def update(self, w, b, dw, db, layer_idx):
        return (w - self.__alpha * dw), (b - self.__alpha * db)


class MomentumUpd(Update):

    def __init__(self, alpha, beta, layer_sizes):
        super().__init__("momentum", True)
        assert (beta > 0) & (beta < 1)
        self.__beta = beta
        self.__alpha = alpha
        self.__other = 1 - beta
        self.__pow_beta = 1
        self.__t_max = int(1 / (1 - beta))
        self.__t = 0
        self.__vw = []
        self.__vb = []
        for i in range(1, len(layer_sizes)):
            self.__vw.append(np.zeros((layer_sizes[i], layer_sizes[i - 1])))
            self.__vb.append(np.zeros((layer_sizes[i], 1)))

    def next_it(self, alpha = 0):
        if alpha != 0:
            self.__alpha = alpha

        if self.__t < self.__t_max:
            self.__t += 1
            self.__pow_beta *= self.__beta

    def reset(self):
        self.__t = 0
        self.__pow_beta = 1
        for i in range(1, len(self.__vw)):
            self.__vw[i - 1] = np.zeros(self.__vw[i - 1].shape)
            self.__vb[i - 1] = np.zeros(self.__vb[i - 1].shape)

    def update(self, w, b, dw, db, layer_idx):
        self.__vw[layer_idx - 1] = vw = self.__vw[layer_idx - 1] * self.__beta + self.__other * dw
        self.__vb[layer_idx - 1] = vb = self.__vb[layer_idx - 1] * self.__beta + self.__other * db

        if self.__t < self.__t_max:
            w_mod = w - self.__alpha * vw / (1 - self.__pow_beta)
            b_mod = b - self.__alpha * vb / (1 - self.__pow_beta)
        else:
            w_mod = w - self.__alpha * vw
            b_mod = b - self.__alpha * vb

        return w_mod, b_mod
