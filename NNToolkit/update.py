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

    def next_it(self, alpha):
        pass

    def reset(self):
        pass

    def update(self, w, b, dw, db, layer_idx):
        return w, b


class StandardUpd(Update):

    def __init__(self, alpha):
        super().__init__("standard", True)
        self.__alpha = alpha

    def next_it(self, alpha=0):
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

    def next_it(self, alpha=0):
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


class RMSPropUpd(Update):

    def __init__(self, alpha, beta, epsilon, layer_sizes):
        super().__init__("momentum", True)
        assert (beta > 0) & (beta < 1)
        self.__epsilon = epsilon
        self.__beta = beta
        self.__alpha = alpha
        self.__other = 1 - beta
        self.__pow_beta = 1
        self.__t_max = int(1 / (1 - beta))
        self.__t = 0
        self.__sw = []
        self.__sb = []
        for i in range(1, len(layer_sizes)):
            self.__sw.append(np.zeros((layer_sizes[i], layer_sizes[i - 1])))
            self.__sb.append(np.zeros((layer_sizes[i], 1)))

    def next_it(self, alpha=0):
        if alpha != 0:
            self.__alpha = alpha

        if self.__t < self.__t_max:
            self.__t += 1
            self.__pow_beta *= self.__beta

    def reset(self):
        self.__t = 0
        self.__pow_beta = 1
        for i in range(1, len(self.__sw)):
            self.__sw[i - 1] = np.zeros(self.__sw[i - 1].shape)
            self.__sb[i - 1] = np.zeros(self.__sb[i - 1].shape)

    def update(self, w, b, dw, db, layer_idx):
        self.__sw[layer_idx - 1] = sw = self.__sw[layer_idx - 1] * self.__beta + self.__other * np.square(dw)
        self.__sb[layer_idx - 1] = sb = self.__sb[layer_idx - 1] * self.__beta + self.__other * np.square(db)

        if self.__t < self.__t_max:
            w_mod = w - self.__alpha * np.divide(dw, np.sqrt(sw / (1 - self.__pow_beta)) + self.__epsilon)
            b_mod = b - self.__alpha * np.divide(db, np.sqrt(sb / (1 - self.__pow_beta)) + self.__epsilon)
        else:
            w_mod = w - self.__alpha * np.divide(dw, np.sqrt(sw) + self.__epsilon)
            b_mod = b - self.__alpha * np.divide(db, np.sqrt(sb) + self.__epsilon)

        return w_mod, b_mod


class AdamUpd(Update):

    def __init__(self, alpha, beta1, beta2, epsilon, layer_sizes):
        super().__init__("momentum", True)
        assert (beta1 > 0) & (beta1 < 1)
        assert (beta2 > 0) & (beta2 < 1)
        self.__epsilon = epsilon
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__alpha = alpha
        self.__other1 = 1 - beta1
        self.__other2 = 1 - beta2
        self.__pow_beta1 = 1
        self.__pow_beta2 = 1
        self.__t_max = max(int(1 / (1 - beta1)),int(1 / (1 - beta1)))
        self.__t = 0
        self.__vw = []
        self.__vb = []
        self.__sw = []
        self.__sb = []
        for i in range(1, len(layer_sizes)):
            self.__vw.append(np.zeros((layer_sizes[i], layer_sizes[i - 1])))
            self.__vb.append(np.zeros((layer_sizes[i], 1)))
            self.__sw.append(np.zeros((layer_sizes[i], layer_sizes[i - 1])))
            self.__sb.append(np.zeros((layer_sizes[i], 1)))

    def next_it(self, alpha=0):
        if alpha != 0:
            self.__alpha = alpha

        if self.__t < self.__t_max:
            self.__t += 1
            self.__pow_beta1 *= self.__beta1
            self.__pow_beta2 *= self.__beta2

    def reset(self):
        self.__t = 0
        self.__pow_beta1 = 1
        self.__pow_beta2 = 1

        for i in range(1, len(self.__sw)):
            self.__vw[i - 1] = np.zeros(self.__vw[i - 1].shape)
            self.__vb[i - 1] = np.zeros(self.__vb[i - 1].shape)
            self.__sw[i - 1] = np.zeros(self.__sw[i - 1].shape)
            self.__sb[i - 1] = np.zeros(self.__sb[i - 1].shape)

    def update(self, w, b, dw, db, layer_idx):
        self.__vw[layer_idx - 1] = vw = self.__vw[layer_idx - 1] * self.__other1 + dw
        self.__vb[layer_idx - 1] = vb = self.__vb[layer_idx - 1] * self.__other1 + db

        self.__sw[layer_idx - 1] = sw = self.__sw[layer_idx - 1] * self.__beta2 + self.__other2 * np.square(dw)
        self.__sb[layer_idx - 1] = sb = self.__sb[layer_idx - 1] * self.__beta2 + self.__other2 * np.square(db)

        if self.__t < self.__t_max:
            w_mod = w - self.__alpha * np.divide(vw / (1 + self.__pow_beta1), np.sqrt(sw / (1 - self.__pow_beta2)) + self.__epsilon)
            b_mod = b - self.__alpha * np.divide(vb / (1 + self.__pow_beta1), np.sqrt(sb / (1 - self.__pow_beta2)) + self.__epsilon)
        else:
            w_mod = w - self.__alpha * np.divide(vw, np.sqrt(sw) + self.__epsilon)
            b_mod = b - self.__alpha * np.divide(vb, np.sqrt(sb) + self.__epsilon)

        return w_mod, b_mod
