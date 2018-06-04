import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import print_matrix
from NNToolkit.manage import learn, create, evaluate, check_gradient
from NNToolkit.util import save_params
from NNToolkit.util import read_params
from NNToolkit.parameters.setup import SetupParams
# from NNToolkit.parameters.result import ResultParams


def init_mean_gt(m = 1000, it = 2000):
    np.random.seed(1)
    parameters = SetupParams()

    threshold = 20
    offset = 10
    size = 10


    parameters.alpha = 0.02
                  #"alpha_min": 0.005,
    parameters.verbosity = 0
    parameters.iterations = it
    parameters.lambd =  8
    parameters.graph = True
    parameters.topology = [size,int(size/2),1]
    parameters.activations = [act.ReLU,act.Sigmoid]  # [act.TanH, act.Sigmoid]

#    width = size - 1

#    while width > 1:
        # for i in range(0,2):
#        parameters["topology"].append(width)
#        if width % 2 == 1:
#            width -= 1
#        else:
#            width = int(width/2)
#    parameters["topology"].append(1)

    print("topology:" + str(parameters.topology))

    parameters.x = np.random.randn(size,m)
    parameters.y = np.int64((np.sum(parameters.x,axis=0,keepdims=True) / size) > 0)
    parameters.x_cv = np.random.randn(size, int(m/5))
    parameters.y_cv = np.int64((np.sum(parameters.x_cv, axis=0, keepdims=True) / size) > 0)

    if parameters.verbosity > 1:
        print("X:  " + print_matrix(parameters.x,4))
        print("Y:  " + print_matrix(parameters.y, 4))

    # save_json_params(parameters,"testCases/sum_of_int_gt.pkl")
    return parameters

def train_gt_mean(n = 1000):
    params = init_mean_gt(n)
    network = learn(params)
    network.get_weights(params.params)
    if params.x_cv is not None:
        y_hat,err = evaluate(network,params.x_cv,params.y_cv)
        tag = 'ecv' + '{:05.2f}'.format(err*100)
    else:
        y_hat, err = evaluate(network, params.x, params.y)
        tag = 'etr' + '{:05.2f}'.format(err * 100)

    save_params(params.to_dict(), "../testCases/mean_gt_" + str(params) + tag + ".json.gz")


def restore_gt_mean(n = 1000):
    params = read_params("../testCases/mean_gt_" + str(n) + ".json.gz")
    network = create(params)
    y_hat, acc = evaluate(network, params["X"], params["Y"])
    print("training accuracy:" + str(acc*100) + "%")

    y_hat, acc = evaluate(network,params["X_t"],params["Y_t"])
    print("test accuracy:    " + str(acc*100) + "%")


def do_check_gradient(n = 1000):
    params = init_mean_gt(n,50)
    check_gradient(params)



train_gt_mean(2000)
# do_check_gradient(1000)
# restore_gt_mean()



