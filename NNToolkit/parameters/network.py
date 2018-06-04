import re
import numpy as np


class NetworkParams:
    __pattern = re.compile("^(w|b|dw|db)\d+$")

    def __init__(self, layer_idx=0, w=None, b=None, dw=None, db=None):
        self.__params = {}
        if layer_idx:
            self.set_params(layer_idx, w, b, dw, db)

    def set_params(self, layer_idx, w, b, dw=None, db=None):
        l_str = str(layer_idx)
        self.__params["w" + l_str] = w
        self.__params["b" + l_str] = b
        if dw & db:
            self.__params["dw" + l_str] = dw
            self.__params["db" + l_str] = db

    def has_params(self, layer_idx):
        l_str = str(layer_idx)
        return (("w" + l_str) in self.__params) & (("b" + l_str) in self.__params)

    def get_params(self, layer_idx):
        l_str = str(layer_idx)
        return self.__params["w" + l_str], self.__params["b" + l_str]

    def set_derivatives(self, layer_idx, dw, db):
        l_str = str(layer_idx)
        self.__params["dw" + l_str] = dw
        self.__params["db" + l_str] = db

    def has_derivatives(self, layer_idx):
        l_str = str(layer_idx)
        return (("dw" + l_str) in self.__params) & (("db" + l_str) in self.__params)

    def get_derivatives(self, layer_idx):
        l_str = str(layer_idx)
        return self.__params["dw" + l_str], self.__params["db" + l_str]

    def is_empty(self):
        return len(self.__params) == 0

    def to_dict(self):
        as_dict = {}
        for key in self.__params:
            match = NetworkParams.__pattern.fullmatch(key)
            if match is not None:
                as_dict[key] = self.__params[key].tolist()
        return as_dict

    @staticmethod
    def from_dict(as_dict):
        params = NetworkParams()
        for key in as_dict:
            match = NetworkParams.__pattern.fullmatch(key)
            if match is not None:
                params.__params[key] = np.array(as_dict[key])
        return params
