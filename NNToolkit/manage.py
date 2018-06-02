import numpy as np
import datetime
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
      "activations": [act.Sigmoid],    # activation function either 2 or one for each layer
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
        print("make_activations")
        if layers > 2:
            act_list = []
            for i in range(1,layers - 1):
                act_list.append(first)
            act_list.append(last)
            return act_list
        else:
            return [last]


    assert "topology" in parameters
    layer_sizes = parameters["topology"]
    layers = len(layer_sizes)
    assert layers > 1

    # add some defaults for optional parameters

    print("before layers:" + str(layers) + " activations:" + str(parameters["activations"]))
    if "activations" not in parameters:
        activations = make_activations(act.TanH,act.Sigmoid,layers)
    else:
        activations = parameters["activations"]
        if len(activations) < (layers - 1):
            activations = make_activations(activations[0],activations[1],layers)

    print("after layers:" + str(layers) + " activations:" + str(activations))

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

def get_error(y_hat,y):
    err = np.linalg.norm(y_hat - y, axis=0, keepdims=True)
    return np.squeeze(np.dot(err, err.T)) / y.shape[1]

def evaluate(network, x, y = None):
    # print("network:\n" + str(network) + "\n")
    res = network.process(x,{})
    acc = None
    if y is not None:
        acc = 1 - get_error(res["Y_hat"], y)

    return res["Y_hat"],acc


def learn(parameters):
    # TODO: timestamps on output
    network = create(parameters)
    print("network:\n" + str(network) + "\n")

    x = parameters["X"]
    y = parameters["Y"]

    verbose = parameters["verbose"]
    iterations = parameters["iterations"]

    graph = False
    if ("graph" in parameters):
        graph = parameters["graph"]
        if graph:
            graph_x = []
            graph_j = []
            graph_e = []

    if parameters["verbose"] > 0:
        print("X:    " + print_matrix(parameters["X"],6))
        print("Y:    " + print_matrix(parameters["Y"],6))

    alpha_min = -1
    alpha = 0

    if "alpha" in parameters:
        alpha = parameters["alpha"]
        if "alpha_min" in parameters:
            alpha_min = parameters["alpha_min"]
            if alpha_min >= alpha:
                alpha_min = -1

    params = {"Y": y, "backprop": True, "alpha": alpha }

    if verbose > 2:
        params["verbose"] = True

    for i in range(0,iterations):
        if (i % (iterations / 100)) == 0:
            by_hundred = True

            if (alpha != 0) & (alpha_min > 0):
                params["alpha"] = adapt_lr(alpha, alpha_min, iterations, i)

            if (i % (iterations / 10)) == 0:
                by_ten = True

                if verbose > 1:
                    params["verbose"] = True
                    print("iteration: " + str(i))
            else:
                by_ten = False
        else:
            by_hundred = False
            by_ten = False

        res = network.process(x, params)

        if by_ten:
            if i > 0:
                err = get_error(res["Y_hat"],y)
            else:
                err = 0

            print("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()) +  " {:5d}".format(i) +
                  " - cost:" + "{:8.5f}".format(res["cost"]) + " err:" + "{:2d}".format(int(err*100)) + "%")

        if by_hundred & graph:
            graph_x.append(i)
            graph_j.append(res["cost"])
            graph_e.append(get_error(res["Y_hat"],y))


        if verbose > 0:
            # TODO: cleanup
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

    if "X_t" in parameters:
        y_hat, acc = evaluate(network,parameters["X_t"],parameters["Y_t"])
        print("test accuracy:    " + str(acc*100) + "%")

    return network
