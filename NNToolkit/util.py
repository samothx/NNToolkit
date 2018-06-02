import numpy as np
import json
import gzip
import re

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
    out = {"np_ndarrays": []}
    for key in parameters:
        value = parameters[key]
        if key == "activations":
            act_names = []
            for act_obj in value:
                act_names.append(str(act_obj))
            # print("act_names" + str(act_names))
            out["activations"] = act_names
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
            out["np_ndarrays"].append(key)
        else:
            out[key] = value

    dump = json.dumps(out)

    if zipped is True:
        with gzip.open(filename, 'wb') as f:
            f.write(dump.encode('utf-8'))
    else:
        f = open(filename, "w")
        f.write(dump)
        f.close()


def read_params(filename, zipped=True):
    # recover parameters from json file / gzipped json file

    def import_class(cname):
        components = cname.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    if zipped is True:
        with gzip.open(filename, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        f = open(filename, "r")
        data = json.load(f)

    # print("loaded:\n" + str(data))

    if "np_ndarrays" in data:
        np_ndarrays = data["np_ndarrays"]
        del data["np_ndarrays"]
    else:
        np_ndarrays = []

    out = {}
    for key in data:
        value = data[key]
        if key == "activations":
            # sample <class 'NNToolkit.activation.TanH'>
            pattern = re.compile("^<class '([^']+)'>$")
            activations = []
            for act_name in value:
                match = pattern.fullmatch(act_name)
                assert match is not None
                name = match.group(1)
                activations.append(import_class(name))
            out["activations"] = activations
        elif key in np_ndarrays:
            out[key] = np.array(value)
        else:
            out[key] = value
    # print("out:" + str(out))
    return out


def divide2sets(x, y, cv_frac, test_frac, shuffle=False, transposed=False):

    assert (cv_frac < 1) & (test_frac < 1) & ((cv_frac + test_frac) < 1)

    # print("x:     " + print_matrix(x,7))
    # print("y:     " + print_matrix(y,7))

    if not transposed:
        # x/y vectors are column vectors
        m = x.shape[1]
        n = x.shape[0]
        n_y = y.shape[0]

        # print("m,n:(" + str(m) + "," + str(n) + ")")
        # print("shape x:" + str(x.shape))
        if shuffle:
            work = np.zeros((m, n + n_y))
            work[:, 0:n] = x.T
            work[:, n:n + n_y] = y.T
            # print("work:  " + print_matrix(work,7))
            np.random.shuffle(work)
            x_work = work[:, 0:n].T
            y_work = work[:, n:n + n_y].T
        else:
            x_work = x
            y_work = y
    else:
        # x/y vectors are row vectors
        m = x.shape[0]
        n = x.shape[1]
        n_y = y.shape[1]
        if shuffle:
            work = np.zeros((m, n + n_y))
            work[:, 0:n] = x
            work[:, n:n + n_y] = y
            np.random.shuffle(work)
            x_work = work[:, 0:n].T
            y_work = work[:, n:n + n_y].T
        else:
            x_work = x.T
            y_work = y.T

    # print("x_work:" + print_matrix(x_work,7))
    # print("y_work:" + print_matrix(y_work,7))

    train_frac = 1 - cv_frac - test_frac
    train_size = int(m * train_frac)

    start = 0
    end = train_size
    x_train = x_work[:, start:end]
    y_train = y_work[:, start:end]

    start = end

    if cv_frac:
        size = int(m * cv_frac)
        if (size == 0) & (train_size < m):
            size = 1
        end = start + size
        x_cv = x_work[:, start:end]
        y_cv = y_work[:, start:end]
        start = end
    else:
        x_cv = None
        y_cv = None

    if (test_frac > 0) & (start < m):
        x_test = x_work[:, start:m]
        y_test = y_work[:, start:m]
    else:
        x_test = None
        y_test = None

    # print("x_train:" + print_matrix(x_train,8))
    # print("y_train:" + print_matrix(y_train,8))

    # if x_cv is not None:
    #     print("x_cv:   " + print_matrix(x_cv,8))
    #     print("y_cv    :" + print_matrix(y_cv,8))

    # if x_test is not None:
    #     print("x_test:  " + print_matrix(x_test,8))
    #     print("y_test:  " + print_matrix(y_test,8))

    res = {"X_train": x_train, "Y_train": y_train}

    if x_cv is not None:
        res["X_cv"] = x_cv
        res["Y_cv"] = y_cv

    if x_test is not None:
        res["X_test"] = x_test
        res["Y_test"] = y_test

    return res
