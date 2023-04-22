import numpy as np
import time


def grad_f(t, x1, x2):
    return np.array([[t * (2 * x1 - x2 - 1) + 3 / (6 - 3 * x1 - 2 * x2) - 1 / x1],
                     [t * (-1 * x1 + 4 * x2 - 10) + 2 / (6 - 3 * x1 - 2 * x2) - 1 / x2]])


def Hessian_f(t, x1, x2):
    return np.array(
        [[2 * t - 9 / pow((6 - 3 * x1 - 2 * x2), 2) + 1 / pow(x1, 2), -t - 6 / pow((6 - 3 * x1 - 2 * x2), 2)],
         [-t - 6 / pow((6 - 3 * x1 - 2 * x2), 2), 4 * t - 4 / pow((6 - 3 * x1 - 2 * x2), 2) + 1 / pow(x2, 2)]])


def NewtonRaphson(t, x1, x2):
    gf = grad_f(t, x1, x2)

    Hf = Hessian_f(t, x1, x2)
    Hf_inv = np.linalg.inv(Hf)

    deltaX = 0.1 * np.matmul(Hf_inv, gf)
    res = np.linalg.norm(deltaX, 2)

    return x1 - deltaX[0, 0], x2 - deltaX[1, 0], res


if __name__ == "__main__":
    time_start = time.time()
    t = 2
    x1 = 1
    x2 = 2
    while True:
        while True:
            x1, x2, res = NewtonRaphson(t, x1, x2)
            if res < 0.0001:
                break

            # print(x1, x2, res)
            # print("------")

        if 3.0 / t < 0.0001:
            time_end = time.time()
            print('consume time:', time_end - time_start)
            print("t:{}, x1:{}, x2:{}".format(t, x1, x2))
            break
        t = 2 * t
