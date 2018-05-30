import numpy as np



def print_matrix(matrix,offset = 0):
    n = matrix.shape[0]
    m = matrix.shape[1]
    pad = ' '*(offset + 1);

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
