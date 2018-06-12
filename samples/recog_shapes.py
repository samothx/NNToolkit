import sys
sys.path.append('../')

from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import math
import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import divide2sets, save_params
from NNToolkit.manage import learn,evaluate
from NNToolkit.parameters.setup import SetupParams
# import time
# import random


def draw_cross(size,offset,color, bg_color, line_width):
    center = (size / 2)
    img = Image.new('RGB', (size, size),color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.line(((center, offset), (center, size - offset)), color, line_width)
    draw.line(((offset, center), (size - offset, center)), color, line_width)
    del draw
    return img

def draw_square(size,offset,color, bg_color, line_width):
    # center = (size / 2)
    points = [(offset, offset),(offset, size - offset), (size - offset, size - offset),(size - offset,offset)]
    img = Image.new('RGB', (size, size),color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.line((points[0], points[1]), color, line_width)
    draw.line((points[1], points[2]), color, line_width)
    draw.line((points[2], points[3]), color, line_width)
    draw.line((points[3], points[0]), color, line_width)
    del draw
    return img

def draw_circle(size,offset,color, bg_color, line_width):
    points0 = ((offset, offset),(size - offset, size - offset))
    points1 = ((offset + line_width, offset + line_width),(size - offset - line_width, size - offset - line_width))

    img = Image.new('RGB', (size, size),color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.ellipse(points0, color, line_width)
    draw.ellipse(points1, bg_color, line_width)
    del draw
    return img

def draw_triangle(size,offset,color, bg_color, line_width):
    len = size - 2 * offset
    height = int(len * math.sqrt(0.75))
    v_offset = (size - height) / 2

    points = ((size / 2,v_offset),(size - offset, size - v_offset),(offset,size - v_offset))
    img = Image.new('RGB', (size, size),color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.line((points[0],points[1]), color, line_width)
    draw.line((points[1], points[2]), color, line_width)
    draw.line((points[2], points[0]), color, line_width)
    del draw
    return img

def add_shape(index,img,x,y,type):
    # print("add index:" + str(index) + " type:" + str(type))
    (m,w,h,d) = x.shape
    tmp_img = img.filter(ImageFilter.GaussianBlur(1))
    x[index,:,:,:] = np.array(tmp_img.getdata()).reshape(w,h,d) / 255
    y[type,index] = 1
    index = index + 1
    tmp_img = img.filter(ImageFilter.GaussianBlur(2))
    x[index,:,:,:] = np.array(tmp_img.getdata()).reshape(w,h,d) / 255
    y[type,index] = 1
    index = index + 1
    return index


def make_shapes(size=32, col_levels=3, line_width=3, offset=6, show=False):
    types = 4
    depth = 3

    assert (col_levels > 1) & (col_levels < 256)
    levels = [0]
    if col_levels > 2:
        for i in range(1,col_levels - 1):
            levels.append(int(i * 256/(col_levels - 1)))
    levels.append(255)

    colors = []

    for i in levels:
        for j in levels:
            for k in levels:
                colors.append((i,j,k))

    num_colors = len(colors)
    # color combinations * blur * shapes
    combinations = num_colors * (num_colors - 1) * 2 * types

    x_train = np.zeros((combinations,size,size,depth),np.float32)
    y_train = np.zeros((4,combinations),np.float32)

    # print(str(pics.shape))

    print("colors:" + str(num_colors) + " combinations:" + str(combinations))

    index = 0
    for fg in colors:
        for bg in colors:
            if not fg == bg:
                # print("fg:" + str(fg) + " bg:" + str(bg))
                img = draw_square(size, offset, fg, bg,line_width)
                index = add_shape(index,img,x_train,y_train,0)
                img = draw_cross(size, offset, fg, bg,line_width)
                index = add_shape(index,img,x_train,y_train,1)
                img = draw_circle(size, offset, fg, bg, line_width)
                index = add_shape(index,img,x_train,y_train,2)
                img = draw_triangle(size, offset, fg, bg, line_width)
                index = add_shape(index,img,x_train,y_train,3)

    if show:
        num_pix = 1024
        num_cols = int(num_pix / size)
        num_pics = int(math.pow(num_cols,2))

        print("num pics:" + str(num_pics))

        if combinations < num_pics:
            num_pics = combinations

        num_rows = int(num_pics / num_cols)
        if num_pics % num_cols:
            num_rows += 1

        big_pic = np.ones((num_rows * size,num_cols * size,depth))
        for i in range(0,num_pics):
            h_start = (i % num_cols) * size
            h_end = h_start + size
            v_start = int(i / num_cols) * size
            v_end = v_start + size
            big_pic[v_start:v_end,h_start:h_end,:] = x_train[i,:,:,:]

        plt.imshow(big_pic)
        plt.show()

    return  x_train, y_train


def init_shape_recog():
    np.random.seed(1)

    x_raw, y_raw = make_shapes(32,3,3,6)

    x_raw = x_raw.reshape(x_raw.shape[0],-1).T
    print(x_raw.shape)

    m = x_raw.shape[1]
    n_0 = x_raw.shape[0]
    n_l = y_raw.shape[0]

#     y_class = np.zeros((m, n_l))
#     for i in range(0,n_l):
#         y_class[:, i:i+1] = np.int64(y_raw == (i + 1))

    res = divide2sets(x_raw, y_raw, 0.05, 0, True, False)

    print("shape: n0:" + str(n_0) + " nL:" + str(n_l) + " m:" + str(m))

    parameters = SetupParams()
    # parameters.check_overflow = True
    parameters.alpha = 0.01           # learn rate
    parameters.alpha_min = 0.005       # use adaptive learn rate this is the value after max iterations/epochs
    parameters.beta1 = 0.95             # parameter for momentum/adam optimizer - switches on momentum
    # parameters.beta2 = 0.999            # parameter for adam optimizer. if this parameters and beta1 is given adam optimizer is used
    parameters.lambd = 3                # L2 regularization parameter
    parameters.keep_prob = 1            # Dropout regularization parameter(1-> no dropout)
    parameters.iterations = 300         # Max number of iterations / epochs
    parameters.graph = True             # Collect data for and display graph whe finished
    parameters.topology = [n_0, 1000,200, n_l]       # layer sizes
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

parameters = init_shape_recog()        # initialize
network = learn(parameters)             # learn

network.get_weights(parameters.params)
file = '../testCases/shapeRecog' + str(parameters)
if not parameters.x_cv is None:
    y_hat,err = evaluate(network, parameters.x_cv, parameters.y_cv)
    file += 'ecv' + '{:5.2f}'.format(err * 100)
else:
    y_hat,err = evaluate(network, parameters.x, parameters.y)
    file += 'etr' + '{:5.2f}'.format(err * 100)

file += ".json.gz"

save_params(parameters.to_dict(),file)



