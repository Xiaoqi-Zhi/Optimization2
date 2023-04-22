import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2022)  # set a constant seed to get samerandom matrixs
A = np.random.rand(500, 100)
x_ = np.zeros([100, 1])
x_[:5, 0] += np.array([i + 1 for i in range(5)])  # x_denotes expected x
b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1  # add a noise to b
lam = 10  # try some different values in {0.1, 1, 10}


def fx(A, x, b, mu):
    f = 1 / 2 * np.linalg.norm(A @ x - b, ord=2) ** 2 + mu * np.linalg.norm(x, ord=1)
    return f


def Beta(A):
    return max(np.linalg.eig(A.T @ A)[0])


def z(A, x, b):
    beta = Beta(A)
    z = (np.eye(len(x)) - A.T @ A / beta) @ x + A.T @ b / beta
    return z


def xp(z, mu, A):
    temp = abs(z) - mu / Beta(A)
    for i in range(len(temp)):
        if temp[i] > 0:
            temp[i] = temp[i]
        else:
            temp[i] = 0
    xp = np.sign(z) * temp
    return xp


def prox(A, x, b, mu, ml):
    k = 0
    fmin = fx(A, x, b, mu)
    fk = fmin
    f_list = [fk]
    while k < ml:
        k = k + 1
        x = xp(z(A, x, b), mu, A)
        fk = fx(A, x, b, mu)
        f_list.append(fk)
        if fk < fmin:
            fmin = fk
    plt.scatter(list(range(len(f_list))), f_list, s=5, color="red")
    plt.show()
    print("迭代结果为：", fmin)


# prox(A,x_,b,lam,100)
# prox(A,x_,b,1,1000)
# prox(A,x_,b,0.1,1000)
def BCD(A, x, b, mu):
    k = 0
    y = np.ones([100, 1])
    fk = fx(A, x, b, mu)
    f_list = [fk]
    while k < 100:
        y = x
        k = k + 1
        for i in range(len(x)):
            if x[i][0] > 0:
                x[i][0] = 1 / (A[:, i].T @ A[:, i]) * (A[:, i].T @ b2(A, x, b, i) - mu)
            elif x[i][0] < 0:
                x[i][0] = 1 / (A[:, i].T @ A[:, i]) * (A[:, i].T @ b2(A, x, b, i) + mu)
            elif abs(A[:, i].T @ b2(A, x, b, i)) <= mu:
                x[i][0] = 0
        fk = fx(A, x, b, mu)
        f_list.append(fk)
    plt.scatter(list(range(len(f_list))), f_list, s=5)
    plt.show()
    print("迭代结果为：", fx(A, x, b, mu))


def b2(A, x, b, n):
    sum = np.zeros((500, 1))
    for i in range(n):
        sum = sum + x[i][0] * A[:, i]
    for i in range(n + 1, len(x)):
        sum = sum + x[i][0] * A[:, i]
    b2 = b - sum
    return b2


A = np.matrix(A)


# BCD(A, x_, b, 10)
# BCD(A, x_, b, 1)
# BCD(A, x_, b, 0.1)

def fxz(A, x, z, b, lam):
    f = 1 / 2 * np.linalg.norm(A @ x - b, ord=2) ** 2 + lam * np.linalg.norm(z, ord=1)
    return f


def Beta(A):
    return max(np.linalg.eig(A.T @ A)[0])


def xp(z, lam, A):
    temp = abs(z) - lam / Beta(A)
    for i in range(len(temp)):
        if temp[i] > 0:
            temp[i] = temp[i]
        else:
            temp[i] = 0
    xp = np.sign(z) * temp
    return xp


def ADMM(A, x, b, lam):
    mu = np.ones([100, 1])
    rho = Beta(A)
    rho_i = np.identity(A.shape[1]) * rho
    z = x
    k = 0
    F = []
    f = fxz(A, x, z, b, lam)
    while k < 100:
        x = np.linalg.inv(A.T @ A + rho_i) @ (A.T @ b + rho * (z - mu))
        z = xp(x + mu, lam, A)
        mu = mu + x - z
        k = k + 1
        deltaf = (f - fxz(A, x, z, b, lam)) / fxz(A, x, z, b, lam)
        f = fxz(A, x, z, b, lam)
        F.append(f)
    plt.scatter(list(range(0, 100)), F, s=5)
    plt.show()
    print(fxz(A, x, z, b, lam))


np.random.seed(2022)
A = np.random.rand(500, 100)
# ADMM(A, x_, b, 10)
# ADMM(A, x_, b, 1)
# ADMM(A, x_, b, 0.1)
