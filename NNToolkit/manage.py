import datetime
import numpy as np
import matplotlib.pyplot as plt
# import NNToolkit.activation as act
from NNToolkit.parameters.result import SetupParams, LayerParams

from NNToolkit.layer import Layer
from NNToolkit.util import adapt_lr
from NNToolkit.util import print_matrix
from NNToolkit.util import read_params

# TODO: use class based params instead to be able to provide secure defaults


def create(parameters):

    def make_activations(first, last, count):
        if count > 2:
            act_list = []
            for idx in range(1, count - 1):
                act_list.append(first)
            act_list.append(last)
            return act_list
        else:
            return [last]

    assert isinstance(parameters, SetupParams)

    layer_count = len(parameters.topology)
    activations = parameters.activations

    if len(activations) < (layer_count - 1):
        activations = make_activations(activations[0], activations[1], layer_count)

    layer_params = LayerParams()
    layer_params.local_params = parameters.local_params
    layer_params.size = parameters.topology[1]
    layer_params.activation = activations[0]()
    if parameters.local_params & parameters.has_params(1):
        layer_params.weight, layer_params.bias = parameters.get_params(1)
    layer_params.valid()

    root_layer = Layer(layer_params, 1, parameters.topology[0])

    for i in range(2, layer_count):
        layer_params.size = parameters.topology[i]
        layer_params.activation = activations[i - 1]()

        if parameters.local_params & parameters.has_params(i):
            layer_params.weight, layer_params.bias = parameters.get_params(i)

        root_layer.add_layer(layer_params)

    layer_params.prev_size = parameters.topology[layer_count - 1]
    root_layer.add_layer(layer_params, True)
    return root_layer


def from_param_file(filename, zipped=True):
    parameters = read_params(filename, zipped)
    return create(parameters)


def get_error(y_hat, y):
    # TODO: different strategies for classification vs regression
    # this is for classification
    # max will be 1 if different else 0 no need for square error
    # as there are only ones involved
    return np.sum(np.max(y_hat != y, axis=0)) / y.shape[1]

    # err = np.linalg.norm(y_hat - y, 2, axis=0,keepdims=True)
    # err = np.zeros((1,y_hat.shape[1]))
    # err[np.where(y != y_hat)] = 1
    # return np.squeeze(np.dot(err, err.T)) / y.shape[1]


def evaluate(network, x, y=None):
    # print("network:\n" + str(network) + "\n")
    res = network.process(x, {})
    err = None
    if y is not None:
        err = get_error(res["Y_hat"], y)

    return res["Y_hat"], err


def learn(parameters):
    assert isinstance(parameters,SetupParams)
    # TODO: ensure alpha and local_params unless we implement external minimization
    parameters.local_params = True

    network = create(parameters)
    print("network:\n" + str(network) + "\n")

    x = parameters["X"]
    y = parameters["Y"]

    verbose = parameters["verbose"]
    iterations = parameters["iterations"]

    graph = False
    graph_x = []
    graph_j = []
    graph_e = []
    graph_e_t = []

    if "graph" in parameters:
        graph = parameters["graph"]

    if parameters["verbose"] > 0:
        print("X:    " + print_matrix(parameters["X"], 6))
        print("Y:    " + print_matrix(parameters["Y"], 6))

    alpha_min = -1
    alpha = 0

    if "alpha" in parameters:
        alpha = parameters["alpha"]
        if "alpha_min" in parameters:
            alpha_min = parameters["alpha_min"]
            if alpha_min >= alpha:
                alpha_min = -1

    params = {"Y": y, "backprop": True, "alpha": alpha}

    if "lambda" in parameters:
        lambd = parameters["lambda"]
        if lambd:
            params["lambda"] = lambd

    if verbose > 2:
        params["verbose"] = True

    if ("X_t" in parameters) & ("Y_t" in parameters):
        y_t = parameters["Y_t"]
        x_t = parameters["X_t"]
    else:
        y_t = None
        x_t = None

    update_iv = min(max(int(iterations / 100), 10), 100)

    for i in range(0, iterations):

        if (i % update_iv) == 0:
            update = True

            if (alpha != 0) & (alpha_min > 0):
                tmp = adapt_lr(alpha, alpha_min, iterations, i)
                # print("new learn rate:" + str(tmp))
                params["alpha"] = tmp
        else:
            update = False

        if (i % (iterations / 10)) == 0:
            if verbose > 1:
                params["verbose"] = True
                print("iteration: " + str(i))

        res = network.process(x, params)

        if update & graph:
            err = get_error(res["Y_hat"], y)
            graph_x.append(i)
            graph_j.append(res["cost"])
            graph_e.append(err)
            if y_t is not None:
                y_hat, err_t = evaluate(network, x_t, y_t)
                graph_e_t.append(err_t)
                err_str = " test err:" + "{:5.2f}".format(err_t * 100) + "%"
            else:
                err_str = ''

            print("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()) + " {:5d}".format(i) +
                  " - cost:" + "{:8.5f}".format(res["cost"]) + " err:" + "{:5.2f}".format(err*100) + "%" + err_str)

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

    if graph:
        plt.subplot(2, 1, 1)
        plt.plot(graph_x, graph_j, label='Cost')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(graph_x, graph_e, label='Train Error')
        if len(graph_e_t):
            plt.plot(graph_x, graph_e_t, label='Test Error')
        plt.legend()
        plt.show()

    if verbose >= 1:
        params["verbose"] = True

    res = network.process(x, params)

    if "cost" in res:
        print("last -  cost:" + str(res["cost"]))

    y_hat, acc = evaluate(network, x, parameters["Y"])
    print("training error:" + "{:2.2f}".format(acc*100) + "%")

    if "X_t" in parameters:
        y_hat, acc = evaluate(network, parameters["X_t"], parameters["Y_t"])
        print("test error:    " + "{:2.2f}".format(acc*100) + "%")

    return network