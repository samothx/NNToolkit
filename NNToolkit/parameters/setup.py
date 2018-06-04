import re
import numpy as np

from NNToolkit.parameters.network import NetworkParams
import NNToolkit.activation as act


class SetupParams:

    def __init__(self):
        self.alpha = 0.01
        self.alpha_min = 0.01
        self.iterations = 1000
        self.activations = [act.ReLU, act.Sigmoid]
        self.topology = None
        self.verbosity = 0
        self.local_params = True
        self.params = NetworkParams()
        self.x = None
        self.y = None
        self.x_cv = None
        self.y_cv = None
        self.x_t = None
        self.y_t = None

    def valid(self):
        assert self.alpha > 0
        assert self.alpha_min >= self.alpha
        assert self.iterations > 0
        assert self.topology is not None
        assert len(self.topology) > 1

    def to_dict(self):
        as_dict = {"alpha": self.alpha,
                   "alpha_min": self.alpha_min,
                   "iterations": self.iterations,
                   "topology": self.topology,
                   "verbosity": self.verbosity,
                   "local_params": self.local_params}

        tmp_list = []
        for act_name in self.activations:
            tmp_list.append(str(act_name))
        as_dict["activations"] = tmp_list

        if not self.params.is_empty():
            as_dict["params"] = self.params.to_dict()

        if self.x is not None:
            as_dict["x"] = self.x.tolist()

        if self.y is not None:
            as_dict["y"] = self.y.tolist()

        if self.x_cv is not None:
            as_dict["x_cv"] = self.x_cv.tolist()

        if self.y_cv is not None:
            as_dict["y_cv"] = self.y_cv.tolist()

        if self.x_t is not None:
            as_dict["x_t"] = self.x_t.tolist()

        if self.y_cv is not None:
            as_dict["y_t"] = self.y_t.tolist()

        return as_dict

    @staticmethod
    def from_dict(setup_dict):

        def import_class(cname):
            components = cname.split('.')
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod

        setup = SetupParams()

        setup.alpha = setup_dict["alpha"]
        setup.alpha_min = setup_dict["alpha_min"]
        setup.iterations = setup_dict["iterations"]
        setup.topology = setup_dict["topology"]
        setup.verbosity = setup_dict["verbosity"]
        setup.local_params = setup_dict["local_params"]

        pattern = re.compile("^<class '([^']+)'>$")
        setup.activations = []
        for act_name in setup_dict["activations"]:
            match = pattern.fullmatch(act_name)
            assert match is not None
            name = match.group(1)
            setup.activations.append(import_class(name))

        if "x" in setup_dict:
            setup.x = np.array(setup_dict["x"])

        if "y" in setup_dict:
            setup.y = np.array(setup_dict["y"])

        if "x_cv" in setup_dict:
            setup.x_cv = np.array(setup_dict["x_cv"])

        if "y_cv" in setup_dict:
            setup.y_cv = np.array(setup_dict["y_cv"])

        if "x_t" in setup_dict:
            setup.x_t = np.array(setup_dict["x_t"])

        if "y_t" in setup_dict:
            setup.y_t = np.array(setup_dict["y_t"])

        if "params" in setup_dict:
            setup.params = NetworkParams.from_dict(setup_dict["params"])

        return setup
