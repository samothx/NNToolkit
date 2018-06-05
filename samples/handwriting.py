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
    parameters.alpha = 0.5
    parameters.alpha_min = 0.05
    parameters.beta1 = 0.9
    parameters.lambd = 8
    parameters.iterations = 300
    parameters.graph = True
    parameters.topology = [n_0, 300, 200, 100, n_l]
    parameters.activations = [act.ReLU,act.Sigmoid]
    parameters.x = res["X_train"]
    parameters.y = res["Y_train"]

    if "X_cv" in res:
        parameters.x_cv = res["X_cv"]
        parameters.y_cv = res["Y_cv"]

    print("sum y_train:" + str(np.sum(res["Y_train"])))
    print("sum y:" + str(np.sum(y_raw > 0)))
    return parameters


parameters = init_hand_writing()
network = learn(parameters)



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
