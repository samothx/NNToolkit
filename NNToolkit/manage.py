import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
# import NNToolkit.activation as act
# from NNToolkit.parameters.result import ResultParams
from NNToolkit.parameters.setup import SetupParams
from NNToolkit.parameters.layer import LayerParams
from NNToolkit.parameters.runtime import RuntimeParams
from NNToolkit.parameters.result import ResultParams
from NNToolkit.parameters.network import NetworkParams

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
    if parameters.local_params & parameters.params.has_params(1):
        layer_params.weight, layer_params.bias = parameters.params.get_params(1)

    layer_params.valid()

    root_layer = Layer(layer_params, 1, parameters.topology[0])

    for i in range(2, layer_count):
        layer_params.size = parameters.topology[i]
        layer_params.activation = activations[i - 1]()

        if parameters.local_params & parameters.params.has_params(i):
            layer_params.weight, layer_params.bias = parameters.params.get_params(i)

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
    params = RuntimeParams()
    params.set_eval(y)
    res = network.process(x, params)
    assert isinstance(res, ResultParams)
    err = None
    if y is not None:
        err = get_error(res.y_hat, y)

    return res, err


def make_rt_params(parameters):
    assert isinstance(parameters, SetupParams)

    rt_params = RuntimeParams()
    rt_params.set_train(parameters.y)

    if not parameters.params.is_empty():
        rt_params.set_params(parameters.params)

    rt_params.set_alpha(parameters.alpha)

    if parameters.lambd:
        rt_params.set_lambda(parameters.lambd)

    if parameters.verbosity > 2:
        rt_params.set_verbose(True)

    return rt_params

def learn(parameters):
    assert isinstance(parameters, SetupParams)
    # TODO: ensure alpha and local_params unless we implement external minimization
    parameters.local_params = True

    rt_params = make_rt_params(parameters)

    network = create(parameters)
    print("network:\n" + str(network) + "\n")

    graph = parameters.graph
    graph_x = []
    graph_j = []
    graph_e = []
    graph_e_t = []

    if parameters.alpha_min < parameters.alpha:
        adapt_alpha = True
    else:
        adapt_alpha = False

    if parameters.verbosity > 0:
        print("X:    " + print_matrix(parameters.x, 6))
        print("Y:    " + print_matrix(parameters.y, 6))

    iterations = parameters.iterations
    update_iv = min(max(int(iterations / 100), 10), 100)

    x = parameters.x

    update = False
    res = None

    for i in range(0, iterations):

        if (i % update_iv) == 0:
            update = True

            if adapt_alpha:
                rt_params.set_alpha(adapt_lr(parameters.alpha, parameters.alpha_min, iterations, i))
        else:
            update = False

        if (i % (iterations / 10)) == 0:
            if parameters.verbosity > 1:
                rt_params.set_verbose(True)
                print("iteration: " + str(i))

        res = network.process(x, rt_params)

        if update:
            err = get_error(res.y_hat, parameters.y)
            if graph:
                graph_x.append(i)
                graph_j.append(res.cost)
                graph_e.append(err)

            if parameters.x_cv is not None:
                y_hat, err_cv = evaluate(network, parameters.x_cv, parameters.y_cv)
                if graph:
                    graph_e_t.append(err_cv)
                err_str = " test err:" + "{:5.2f}".format(err_cv * 100) + "%"
            else:
                err_str = ''

            print("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()) + " {:5d}".format(i) +
                  " - cost:" + "{:8.5f}".format(res.cost) + " err:" + "{:5.2f}".format(err*100) + "%" + err_str)

        if parameters.verbosity > 0:
            # TODO: cleanup
            if (parameters.verbosity < 3) & rt_params.is_verbose():
                rt_params.set_verbose(False)
            if parameters.verbosity > 1:
                print("Y_hat:" + print_matrix(res.y_hat, 6) + "\n")
                print("Y    :" + print_matrix(parameters.y, 6) + "\n")
            if (parameters.verbosity > 2) | ((i % (iterations / 10)) == 0):
                print("{:5d}".format(i) + " - cost:" + str(res.cost))
                if parameters.verbosity > 1:
                    print("***********************************************")

    if not update & (res is not None):
        err = get_error(res.y_hat, parameters.y)
        if graph:
            graph_x.append(iterations)
            graph_j.append(res.cost)
            graph_e.append(err)

        if parameters.x_cv is not None:
            y_hat, err_cv = evaluate(network, parameters.x_cv, parameters.y_cv)
            if graph:
                graph_e_t.append(err_cv)
            err_str = " test err:" + "{:5.2f}".format(err_cv * 100) + "%"
        else:
            err_str = ''

        print("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()) + " {:5d}".format(iterations) +
              " - cost:" + "{:8.5f}".format(res.cost) + " err:" + "{:5.2f}".format(err*100) + "%" + err_str)

    if parameters.graph:
        plt.subplot(2, 1, 1)
        plt.plot(graph_x, graph_j, label='Cost')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(graph_x, graph_e, label='Train Error')
        if len(graph_e_t):
            plt.plot(graph_x, graph_e_t, label='Test Error')
        plt.legend()
        plt.show()

    if parameters.verbosity >= 1:
        rt_params.set_verbose(True)
        res = network.process(x, rt_params)
        print("last -  cost:" + str(res.cost))

        y_hat, acc = evaluate(network, x, parameters.y)
        print("training error:" + "{:2.2f}".format(acc*100) + "%")

        if "X_t" in parameters:
            y_hat, acc = evaluate(network, parameters.x_cv, parameters.y_cv)
            print("test error:    " + "{:2.2f}".format(acc*100) + "%")

    return network


def check_gradient(parameters):
    assert isinstance(parameters,SetupParams)
    print("checking gradient")
    network = create(parameters)
    rt_params = make_rt_params(parameters)
    res = None
    if parameters.iterations > 0:
        # run defined iterations
        for i in range(0,parameters.iterations):
            res = network.process(parameters.x,rt_params)

    # this is my master copy of the weights
    nw_params = NetworkParams()
    network.get_weights(nw_params)
    print("weights:" + str(nw_params))
    # make network use weights provided in params instead of local weights
    network.set_local_params(False)
    rt_params.set_params(nw_params)
    # no more parameter updates
    rt_params.set_alpha(0)
    res = network.process(parameters.x,rt_params)
    assert isinstance(res,ResultParams)

    print("cost1:" + str(res.cost))

    res_params = res.get_params()
    assert isinstance(res_params,NetworkParams)
    print("grads 0:  " + str(res_params))

    # test network with params provided
    rt_params.set_params(nw_params)
    res = network.process(parameters.x, rt_params)
    print("after: " + str(res.cost))

    layer_params = []

    epsilon = 1e-7
    approx = []
    grad = []

    for l in range(1,len(parameters.topology)):
        print("processing layer:" + str(l))
        assert nw_params.has_params(l) & res_params.has_derivatives(l)
        w,b = nw_params.get_params(l)
        dw,db = res_params.get_derivatives(l)
        w_tmp = np.copy(w)
        nw_params.set_params(l, w_tmp, b)
        rt_params.set_params(nw_params)

        for i in range(0,w.shape[0]):
            for j in range(0,w.shape[1]):
                saved_val = w_tmp[i,j]
                w_tmp[i,j] = saved_val + epsilon
                res_plus = network.process(parameters.x,rt_params)
                w_tmp[i, j] = saved_val - epsilon
                res_minus = network.process(parameters.x,rt_params)
                approx.append((res_plus.cost - res_minus.cost) / (2*epsilon))
                grad.append(dw[i,j])
                w_tmp[i, j] = saved_val

        b_tmp = np.copy(b)
        nw_params.set_params(l, w, b_tmp)
        rt_params.set_params(nw_params)

        for i in range(0, b.shape[0]):
            saved_val = b_tmp[i, 0]
            b_tmp[i, 0] = saved_val + epsilon
            res_plus = network.process(parameters.x, rt_params)
            b_tmp[i, 0] = saved_val - epsilon
            res_minus = network.process(parameters.x, rt_params)
            approx.append((res_plus.cost - res_minus.cost) / (2 * epsilon))
            grad.append(db[i, 0])
            b_tmp[i, 0] = saved_val

        # print("res." + str(i) + ":" + str(len(layer_params[len(layer_params) - 1])))

    print("approx:" + str(len(approx)))
    print("grads: " + str(len(grad)))

    approx = np.array(approx).reshape((1,len(approx)))
    grad = np.array(grad).reshape((1,len(grad)))

    print("approx:" + str(approx))
    print("grad  :" + str(grad))
    err = np.linalg.norm(approx - grad) / (np.linalg.norm(approx) + np.linalg.norm(grad))
    print("error:" + str(err))
    return err






