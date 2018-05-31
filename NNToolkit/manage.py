import numpy as np

import NNToolkit.activation as act

from NNToolkit.terminal import Terminal
from NNToolkit.layer import Layer
from NNToolkit.util import adapt_lr
from NNToolkit.util import print_matrix
from NNToolkit.util import read_params

default_params = {
      "local_params": True,     # initialize & cache W,b in layer
      "alpha": 0.01,            # learning rate alpha - also triggers update of W,b in layers
      "alpha_min": 0.01,        # when < alpha triggers adaptive learn rate
      "verbose": 0,             # levels of verbsity 0-3
      "iterations": 3000,       # how many iterations of grdient descent
      "epsilon": 0.01,          # min/max element size for initialization of W
      "activations": [act.TanH, act.Sigmoid],    # activation function either 2 or one for each layer
      "topology": []            # array of layer sizes
      # "X": [[]]               # training input vectors
      # "Y": [[]]               # training output vectors
      # "X_t" : [[]]            # test input vectors
      # "Y_t" : [[]]            # test output vectors
      # "W": [[]]               # computed weights
      # "b": [[]]               # computes interceptors
}


def create(parameters):
    def make_activations(first,last,layers):
        act_list = []
        for i in range(1,layers - 1):
            act_list.append(first)
        act_list.append(last)
        return act_list

    assert "topology" in parameters
    layer_sizes = parameters["topology"]
    layers = len(layer_sizes)
    assert layers > 0

    # add some defaults for optional parameters

    if "activations" not in parameters:
        activations = make_activations(act.TanH,act.Sigmoid,layers)
    else:
        activations = parameters["activations"]
        if len(activations) < (layers - 1):
            activations = make_activations(activations[0],activations[1],layers)

    print("layers:" + str(layers) + " activations:" + str(len(activations)))

    if "epsilon" not in parameters:
        epsilon = 0.01
    else:
        epsilon = parameters["epsilon"]

    if "local_params" not in parameters:
        local_params = True
    else:
        local_params = parameters["local_params"]

    def_layer_params = { "epsilon": epsilon, "local_params": local_params }
    root = None

    for i in range(0,len(layer_sizes)):
        layer_params = def_layer_params
        layer_params["size"] = layer_sizes[i]

        if i == 0:
            # layer 0 only needs size
            root = Layer(layer_params)
        else:
            # get the activations right
            layer_params["activation"] = activations[i - 1]()
            if local_params:
                name = "W" + str(i)
                if name in parameters:
                    layer_params["W"] = parameters[name]
                    name = "b" + str(i)
                if name in parameters:
                    layer_params["b"] = parameters[name]
            root.add_layer(Layer(layer_params))

    root.add_layer(Terminal())
    return root

def fromParamFile(filename,zip = True):
    parameters = read_params(filename,zip)
    return create(parameters)

def evaluate(network, x, y = None):
    # print("network:\n" + str(network) + "\n")
    res = network.process(x,{})
    acc = None
    if y is not None:
        err = res["Y_hat"] - y
        acc = (1 - np.squeeze(np.dot(err, err.T)) / y.shape[1])

    return res["Y_hat"],acc


def learn(parameters):
    network = create(parameters)
    print("network:\n" + str(network) + "\n")

    x = parameters["X"]
    y = parameters["Y"]
    verbose = parameters["verbose"]
    iterations = parameters["iterations"]

    if parameters["verbose"] > 0:
        print("X:    " + print_matrix(parameters["X"],6))
        print("Y:    " + print_matrix(parameters["Y"],6))

    params = {"Y":y, "backprop":True, "alpha":parameters["alpha"] }

    if verbose > 2:
        params["verbose"] = True

    for i in range(0,iterations):
        if ("alpha_min" in parameters) & (i > 0) & ((i % 100) == 0):
            if parameters["alpha_min"] < parameters["alpha"]:
                params["alpha"] = adapt_lr(parameters["alpha"],parameters["alpha_min"],iterations,i)

        if (verbose > 1) & (((i % (iterations / 10)) == 0)):
            params["verbose"] = True
            print("iteration: " + str(i))
        res = network.process(x, params)
        if (verbose < 3) & ("verbose" in params):
            del params["verbose"]
        if verbose > 1:
            print("Y_hat:" + print_matrix(res["Y_hat"], 6) + "\n")
            print("Y    :" + print_matrix(params["Y"], 6) + "\n")
        if (verbose > 2) | (((i % (iterations / 10)) == 0) & ("cost" in res)):
            print("{:5d}".format(i) + " - cost:" + str(res["cost"]))
            if verbose > 1:
                print("***********************************************")

    if verbose >= 1:
        params["verbose"] = True
    res = network.process(x, params)
    if "cost" in res:
        print("last -  cost:" + str(res["cost"]))

    y_hat, acc = evaluate(network,x,parameters["Y"])
    print("training accuracy:" + str(acc*100) + "%")

    y_hat, acc = evaluate(network,parameters["X_t"],parameters["Y_t"])
    print("test accuracy:    " + str(acc*100) + "%")

    return network
