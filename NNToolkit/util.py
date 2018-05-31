import numpy as np
# import pickle
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
def print_matrix(matrix,padding = 0):
    n = matrix.shape[0]
    m = matrix.shape[1]
    pad = ' '*(padding + 1);

    res = '['
    for i in range(0, n):
        if i > 0:
            res += ",\n" + pad
        res += "["
        for j in range(0, m):
            res = res + "{:10.5f}".format(float(matrix[i, j])) + ' '
        res += "]"
    res += "]"
    return res

# save parameters to json file / gzipped json file

def save_params(parameters, filename, zip=True):
    out = { "np_ndarrays" : [] }
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

    if zip is True:
        with gzip.open(filename, 'wb') as f:
            f.write(dump.encode('utf-8'))
    else:
        f = open(filename, "w")
        f.write(dump)
        f.close()

# recover parameters from json file / gzipped json file

def read_params(filename,zip = True):
    def import_class(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    data = ''
    if zip is True:
        with gzip.open(filename,"rb") as f:
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

