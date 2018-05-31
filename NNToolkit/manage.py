import numpy as np

import NNToolkit.activation as act

from NNToolkit.terminal import Terminal
from NNToolkit.layer import Layer
from NNToolkit.util import adapt_lr
from NNToolkit.util import print_matrix

default_params = {
      "local_params": True,     # initialize & cache W,b in layer
      "alpha" : 0.01,           # learning rate alpha - also triggers update of W,b in layers
      "alpha_min" : 0.01,       # when < alpha triggers adaptive learn rate
      "verbose" : 0,            # levels of verbsity 0-3
      "iterations" : 3000,      # how many iterations of grdient descent
      "epsilon" : 0.01,         # min/max element size for initialization of W
      "activations" : [act.TanH,act.Sigmoid],    # activation function either 2 or one for each layer
      "topology" : []           # array of layer sizes
      # "X": [[]]               # training input vectors
      # "Y": [[]]               # training output vectors
      # "X_t" : [[]]            # test input vectors
      # "Y_t" : [[]]            # test output vectors
    }

def create(parameters):
    assert "topology" in parameters
    layer_sizes = parameters["topology"]
    layers = len(layer_sizes)
    assert layers > 0

    if not "activations" in parameters:
        activations = (act.TanH,act.Sigmoid)
    else:
        activations = parameters["activations"]

    if not "epsilon" in parameters:
        epsilon = 0.01
    else:
        epsilon = parameters["epsilon"]

    if not "local_params" in parameters:
        local_params = True
    else:
        local_params = parameters["local_params"]


    if len(activations) >= layers:
        root = Layer(layer_sizes[0], activations[0], epsilon,local_params)
        for i in range(1, layers):
            root.add_layer(Layer(layer_sizes[i], activations[i],epsilon, local_params))
    else:
        first_act = activations[0]
        last_act = activations[1]
        if layers == 1:
            first_act = activations[1]
        root = Layer(layer_sizes[0], first_act(),epsilon, local_params)
        for i in range(1, layers - 1):
            root.add_layer(Layer(layer_sizes[i], first_act(),epsilon, local_params))
        root.add_layer(Layer(layer_sizes[layers - 1], last_act(), epsilon, local_params))

    root.add_layer(Terminal())
    return root


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

    res = network.process(x,{})
    err = res["Y_hat"] - parameters["Y"]
    acc = (1 - np.squeeze(np.dot(err,err.T))/parameters["Y"].shape[1]) * 100
    print("training accuracy:" + str(acc) + "%")

    res = network.process(parameters["X_t"],{})
    err = res["Y_hat"] - parameters["Y_t"]
    acc = (1 - np.squeeze(np.dot(err,err.T))/parameters["Y_t"].shape[1]) * 100
    print("test accuracy:    " + str(acc) + "%")

    return network
