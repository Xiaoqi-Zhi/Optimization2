import numpy as np
from math import inf


def Termination(target):
    for x in target:
        if x < 0:
            return False
    return True


def index(table):
    min = table[0]
    index2 = 0
    for i, t in enumerate(table):
        if table[i] < min:
            min = table[i]
            index2 = i
    return index2


c = np.array([-5, -1])
b = np.array([[5], [8]])
Info = np.array([[1, 2], [2,1/2]])  # 代表限制条件
basic = np.eye(Info.shape[0])  # 产生基变量
s = np.append(Info, basic, axis=1)
A = np.append(s, b, axis=1)
c1 = np.append(c, np.zeros(Info.shape[0]))
print(c1)
print(A)
Cost = np.array(np.zeros(A.shape[1]))
Cost[0:c.shape[0]] = c
BV = np.array(range(c.shape[0], A.shape[1] - 1))
temp = np.array([Cost[y] for y in BV]).reshape(1, 2)
target = Cost - np.dot(temp, A)

print(target)
target_zero = [x < 0 for x in target[0][:target[0].shape[0] - 1]]

target_zero = [x for x in target[0][:target[0].shape[0] - 1]]
while 1:
    if Termination(target[0]) == True:
        break
    ans = np.zeros(A.shape[0])
    for i in range(0, A.shape[0]):
        ans[i] = inf
    enter = index(target[0][:target[0].shape[0] - 1])
    for i in range(0, A.shape[0]):
        if A[i][enter] > 0:
            ans[i] = A[i][A.shape[1] - 1] / A[i][enter]
        else:
            ans[i] = inf
    leave = BV[index(ans)]
    A[index(ans)] = A[index(ans)] / A[index(ans)][enter]
    for i in range(0, A.shape[0]):
        if i != index(ans):
            A[i] = A[i][:] - A[i][enter] * A[index(ans)]
        else:
            A[i] = A[index(ans)]
    BV[index(ans)] = enter
    res = 0
    print("basis feasible vairable:")
    for i in range(0, A.shape[0]):
        print("x", BV[i], ":")
        res += c1[BV[i]] * A[i][A.shape[1] - 1]
        print(A[i][A.shape[1] - 1])
    print("result:")
    print(res)
    print(A)
    print("target[0]，before:")
    print(target[0])
    temp = np.array([Cost[y] for y in BV]).reshape(1, 2)
    target[0] = target[0] - np.dot(temp, A)
    print(target[0])

