import numpy as np
import json
import gzip
import re

from NNToolkit.parameters.setup import SetupParams

# parameters:
# alpha_max:  max learning rate
# alpha_min:  min learning rate
# i_max:      max number of iterations
# i_curr:     index of current iteration

# adaptive learning rate, so that
# adapt_lr(alpha_max, alpha_min, i_max, 0)      = alpha_max
# adapt_lr(alpha_max, alpha_min, i_max, i_max)  = alpha_min
# idea: calculate a, b so that a/(b + i_curr) = alpha_max for i_curr = 0
#                              a/(b + i_curr) = alpha_min for i_curr = i_max
# cache values of b for tuples (alpha_max, alpha_min, i_max)
alpha_params = {}


def adapt_lr(alpha_max, alpha_min, i_max, i_curr):
    # TODO: probably can do better than this
    if (alpha_max, alpha_min, i_max) in alpha_params:
        b = alpha_params[(alpha_max, alpha_min, i_max)]
    else:
        alpha_params[(alpha_max, alpha_min, i_max)] = b = - i_max / (1 - alpha_max/alpha_min)
    return alpha_max * b / (i_curr + b)

# print a matrix to a string
# line break after each row,
# padded with padding + 1 spaces starting with the second row
def print_matrix(matrix, padding=0):
    # TODO: solution for big matrices
    n = matrix.shape[0]
    m = matrix.shape[1]
    pad = ' '*(padding + 1)

    if n < 30:
        n_0 = n
        n_1 = 0
    else:
        n_0 = 10
        n_1 = n-10

    if m < 30:
        m_0 = m
        m_1 = 0
    else:
        m_0 = 5
        m_1 = m - 5

    res = '['
    for i in range(0, n_0):
        if i > 0:
            res += ",\n" + pad
        res += "["
        for j in range(0, m_0):
            res = res + "{:10.5f}".format(float(matrix[i, j])) + ' '
        if m_1:
            res += ",...,"
            for j in range(m_1, m):
                res = res + "{:10.5f}".format(float(matrix[i, j])) + ' '

        res += "]"

    if n_1:
        res += ",\n" + pad + " ... not displaying " + str(n_1-n_0) + " lines"
        for i in range(n_1, n):
            if i > n_1:
                res += "\n" + pad + "["
            else:
                res += ",\n" + pad + "["
            for j in range(0, m_0):
                res = res + "{:10.5f}".format(float(matrix[i, j])) + ' '

            if m_1:
                res += ",...,"
                for j in range(m_1, m):
                    res = res + "{:10.5f}".format(float(matrix[i, j])) + ' '

            res += "]"

    res += "]"
    return res

# save parameters to json file / gzipped json file


def save_params(parameters, filename, zipped=True):
    dump = json.dumps(parameters)

    if zipped is True:
        with gzip.open(filename, 'wb') as f:
            f.write(dump.encode('utf-8'))
    else:
        f = open(filename, "w")
        f.write(dump)
        f.close()

def read_params(filename, zipped=True):
    # recover parameters from json file / gzipped json file

    if zipped is True:
        with gzip.open(filename, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        f = open(filename, "r")
        data = json.load(f)

    # print("loaded:\n" + str(data))

    return data


def shuffle_xy(x,y):
    m = x.shape[1]
    n = x.shape[0]
    n_y = y.shape[0]

    permutation = list(np.random.permutation(m))
    return x[:, permutation].reshape((n, m)), y[:, permutation].reshape((n_y, m))


def divide2sets(x, y, cv_frac, test_frac, shuffle=False, transposed=False):

    assert (cv_frac < 1) & (test_frac < 1) & ((cv_frac + test_frac) < 1)

    # print("x:     " + print_matrix(x,7))
    # print("y:     " + print_matrix(y,7))

    if transposed:
        x_wrk = x.T
        y_wrk = y.T
    else:
        x_wrk = x
        y_wrk = y

        # x/y vectors are column vectors
    m = x_wrk.shape[1]
    n = x_wrk.shape[0]
    n_y = y_wrk.shape[0]

    # print("m,n:(" + str(m) + "," + str(n) + ")")
    # print("shape x:" + str(x.shape))
    if shuffle:
        x_wrk,y_wrk = shuffle_xy(x_wrk, y_wrk)

    # print("x_wrk:" + print_matrix(x_wrk,7))
    # print("y_wrk:" + print_matrix(y_wrk,7))

    train_frac = 1 - cv_frac - test_frac
    train_size = int(m * train_frac)

    start = 0
    end = train_size
    x_train = x_wrk[:, start:end]
    y_train = y_wrk[:, start:end]

    start = end

    if cv_frac > 0:
        size = int(m * cv_frac)
        if (size == 0) & (train_size < m) & (test_frac == 0):
            size = 1
        end = start + size
        x_cv = x_wrk[:, start:end]
        y_cv = y_wrk[:, start:end]
        start = end
    else:
        x_cv = None
        y_cv = None

    if (test_frac > 0):
        x_test = x_wrk[:, start:m]
        y_test = y_wrk[:, start:m]
    else:
        x_test = None
        y_test = None


    res = {"X_train": x_train, "Y_train": y_train}

    if x_cv is not None:
        res["X_cv"] = x_cv
        res["Y_cv"] = y_cv

    if x_test is not None:
        res["X_test"] = x_test
        res["Y_test"] = y_test

    return res
