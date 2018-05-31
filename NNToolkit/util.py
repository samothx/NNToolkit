import numpy as np

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
