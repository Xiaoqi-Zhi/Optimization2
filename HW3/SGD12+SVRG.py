import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('a9a.csv')
data = data.values
n = data.shape[0]
data2 = []
for i in range(n):
    a = data[i][0].split()
    data2.append(a)
y = []
for i in range(n):
    temp = int(data2[i][0])
    y.append(temp)
y = np.array(y)
y = y.reshape(-1, 1)
c = []
for i in range(n):
    b = []
    for j in range(1, len(data2[i])):
        index = data2[i][j].find(':')
        b.append(int(data2[i][j][0:index]))
    c.append(b)
m = max(max(i) for i in c)
X = np.zeros((n, m))
for i in range(n):
    for j in range(len(c[i])):
        X[i, c[i][j] - 1] = 1
X = np.matrix(X)

import math


def SGD_fix(A, b, lam):
    k = np.random.randint(0, A.shape[0])
    t = 0
    x = np.zeros((A.shape[1], 1))
    F = []
    sum = 0
    for i in range(A.shape[0]):
        sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
    f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
    F.append(f)
    deltaF = 1
    while abs(deltaF / f) > 0.0000001:
        k = np.random.randint(0, A.shape[0])
        x = x - 0.001 * ((-math.exp(-b[k, :][0] * A[k, :] @ x) * b[k, :][0] * A[k, :].T) / (
                1 + math.exp(-b[k, :][0] * A[k, :] @ x)) + 2 * lam * x)
        sum = 0
        for i in range(A.shape[0]):
            sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
        f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
        F.append(f)
        t = t + 1
        deltaF = F[t] - F[t - 1]
        print(deltaF / f, t)
    plt.scatter(list(range(len(F))), F, s=5)
    plt.show()
    print(f)


SGD_fix(X, y, 0.01 / X.shape[0])


def SGD_dr(A, b, lam):
    k = np.random.randint(0, A.shape[0])
    t = 0
    x = np.zeros((A.shape[1], 1))
    F = []
    sum = 0
    for i in range(A.shape[0]):
        sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
    f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
    F.append(f)
    deltaF = 1
    while abs(deltaF / f) > 0.0000001:
        t = t + 1
        k = np.random.randint(0, A.shape[0])
        x = x - 1 / t * ((-math.exp(-b[k, :][0] * A[k, :] @ x) * b[k, :][0] * A[k, :].T) / (
                1 + math.exp(-b[k, :][0] * A[k, :] @ x)) + 2 * lam * x)
        sum = 0
        for i in range(A.shape[0]):
            sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
        f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
        F.append(f)
        deltaF = F[t] - F[t - 1]
        print(deltaF / f, t)
    plt.scatter(list(range(len(F))), F, s=5)
    plt.show()
    print(f)


SGD_dr(X, y, 0.01 / X.shape[0])


def SVRG(A, b, lam):
    t = 0
    x = np.zeros((A.shape[1], 1))
    y = x
    F = []
    sum = 0
    for i in range(A.shape[0]):
        sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
    f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
    F.append(f)
    deltaF = 1
    grad = 0
    for i in range(A.shape[0]):
        grad = grad + (-math.exp(-b[i, :][0] * A[i, :] @ x) * b[i, :][0] * A[i, :].T) / (
                1 + math.exp(-b[i, :][0] * A[i, :] @ x)) + 2 * lam * x
    gradF = grad / A.shape[0]
    T = 0
    while abs(deltaF / f) > 0.00001:
        #     while T<10000:
        xs = []
        t = 0
        grad = 0
        for i in range(A.shape[0]):
            grad = grad + (-math.exp(-b[i, :][0] * A[i, :] @ y) * b[i, :][0] * A[i, :].T) / (
                    1 + math.exp(-b[i, :][0] * A[i, :] @ y)) + 2 * lam * y
        grady = grad / A.shape[0]
        while t < 20:
            t = t + 1
            T = T + 1
            k = np.random.randint(0, A.shape[0])
            gradkx = (-math.exp(-b[k, :][0] * A[k, :] @ x) * b[k, :][0] * A[k, :].T) / (
                    1 + math.exp(-b[k, :][0] * A[k, :] @ x)) + 2 * lam * x
            gradky = (-math.exp(-b[k, :][0] * A[k, :] @ y) * b[k, :][0] * A[k, :].T) / (
                    1 + math.exp(-b[k, :][0] * A[k, :] @ y)) + 2 * lam * y
            v = gradkx - (gradky - grady)
            x = x - 0.01 * v
            xs.append(x)
            sum = 0
            for i in range(A.shape[0]):
                sum = sum + math.log(1 + math.exp(-b[i, :][0] * A[i, :] @ x))
            f = (1 / A.shape[0]) * sum + lam * np.linalg.norm(x, ord=2) ** 2
            F.append(f)
            deltaF = F[T] - F[T - 1]
            print(deltaF / f, T)
        sumx = 0
        for i in range(len(xs)):
            sumx = sumx + xs[i]
        y = sumx / len(xs)
    plt.scatter(list(range(len(F))), F, s=5)
    plt.show()
    print(f)


SVRG(X, y, 0.01 / X.shape[0])
