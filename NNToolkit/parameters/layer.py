import NNToolkit.activation as act


class LayerParams:
    def __init__(self):
        self.local_params = True
        self.activation = None
        self.size = 0
        self.weight = None
        self.bias = None

    def valid(self):
        assert self.size > 0
        assert self.activation is not None
        assert act.Activation in self.activation.__class__.__bases__
