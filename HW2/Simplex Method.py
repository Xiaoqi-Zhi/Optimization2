from scipy import optimize
import numpy as np

c = np.array([-5,-1])
A = np.array([[1,1],[2,0.5]])
B = np.array([5,8])

res = optimize.linprog(c,A,B)
print(res)