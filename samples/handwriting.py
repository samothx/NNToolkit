import sys
sys.path.append('../')
import numpy as np
import scipy.io

import NNToolkit.activation as act
from NNToolkit.util import divide2sets, save_params
from NNToolkit.manage import learn,evaluate
from NNToolkit.parameters.setup import SetupParams

# best set
# 400,300,200.10
# alpha 0.5
# alpha_min 0.05
# epsilon 0.01
# relu,sigmoid
# 92  / 82 % @ 1000

def init_hand_writing():
    np.random.seed(1)
    matrix = scipy.io.loadmat("../data/mlnn.mat")
    assert isinstance(matrix["X"], np.ndarray)
    assert isinstance(matrix["y"], np.ndarray)

    x_raw = matrix["X"]
    y_raw = matrix["y"]

    m = x_raw.shape[0]
    n_0 = x_raw.shape[1]
    n_l = 10

    y_class = np.zeros((m, n_l))
    for i in range(0,n_l):
        y_class[:, i:i+1] = np.int64(y_raw == (i + 1))

    res = divide2sets(x_raw, y_class, 0.01, 0, True, True)

    print("shape: n0:" + str(n_0) + " nL:" + str(n_l) + " m:" + str(m))

    parameters = SetupParams()
    # parameters.check_overflow = True
    parameters.alpha = 0.0007           # learn rate
    parameters.alpha_min = 0.0005       # use adaptive learn rate this is the value after max iterations/epochs
    parameters.beta1 = 0.95             # parameter for momentum/adam optimizer - switches on momentum
    parameters.beta2 = 0.999            # parameter for adam optimizer. if this parameters and beta1 is given adam optimizer is used
    parameters.lambd = 3                # L2 regularization parameter
    parameters.keep_prob = 1            # Dropout regularization parameter(1-> no dropout)
    parameters.iterations = 300         # Max number of iterations / epochs
    parameters.graph = True             # Collect data for and display graph whe finished
    parameters.topology = [n_0, 200,100, n_l]       # layer sizes
    parameters.activations = [act.ReLU,act.Softmax] # activations. If 2 activations are given but more layers present
                                                    # the first activation will be used for all but the last layers
    parameters.x = res["X_train"]       # training data set
    parameters.y = res["Y_train"]       # training labels

    if "X_cv" in res:
        parameters.x_cv = res["X_cv"]   # cross validation data set
        parameters.y_cv = res["Y_cv"]   # cross validation labels

    print("sum y_train:" + str(np.sum(res["Y_train"])))
    print("sum y:" + str(np.sum(y_raw > 0)))
    return parameters


parameters = init_hand_writing()        # initialize
network = learn(parameters)             # learn


network.get_weights(parameters.params)
file = '../testCases/handWr' + str(parameters)
if not parameters.x_cv is None:
    y_hat,err = evaluate(network, parameters.x_cv, parameters.y_cv)
    file += 'ecv' + '{:5.2f}'.format(err * 100)
else:
    y_hat,err = evaluate(network, parameters.x, parameters.y)
    file += 'etr' + '{:5.2f}'.format(err * 100)

file += ".json.gz"


save_params(parameters.to_dict(),file)

#ts = "{:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
#if "X_t" in params:
#    y_hat,acc = evaluate(network, params["X_t"], params["Y_t"])
#    acc_tag = "_" + "{:02d}".format(int(acc * 100))
#else:
#    acc_tag = ""

# save_params(params, "../testCases/handWr_1000_t_92_" + ts + acc_tag + ".json.gz")
