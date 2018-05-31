from NNToolkit.util import adapt_lr


def test_adapt_alpha():
    alpha_max = 0.1
    alpha_min = 0.01
    i_max = 10000
    i_cur = 0

    print(
        "alpha max:" + "{:6.2f}".format(alpha_max) + " min:" + "{:6.2f}".format(alpha_min) + " i max:" + "{:6d}".format(
            i_max) +
        " curr:" + "{:6d}".format(i_cur) + " alpha:" + "{:8.4f}".format(adapt_lr(alpha_max, alpha_min, i_max, i_cur)))

    for i in range(0, 21):
        i_cur = int(i_max * i / 20)
        print("alpha max:" + "{:6.2f}".format(alpha_max) + " min:" + "{:6.2f}".format(
            alpha_min) + " i max:" + "{:6d}".format(i_max) +
              " curr:" + "{:6d}".format(i_cur) + " alpha:" + "{:8.4f}".format(
            adapt_lr(alpha_max, alpha_min, i_max, i_cur)))
