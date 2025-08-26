from __future__ import division,print_function,unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
Z = np.array([[2,1,2]])
ones = np.ones((X.shape[0],1))
Xbar = np.concatenate((ones,X),axis = 1)
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T,y)
w = np.dot(np.linalg.inv(A),b)
w_0 = w[0][0]
w_1 = w[1][0]
xo = np.linspace(145,185,2)
yo = w_0 + xo * w_1
print(xo)
plt.plot(X,y,'ro')
plt.plot(xo,yo)
plt.axis([140,190,40,80])
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
y1 = w_0 + 155 * w_1
rerg = linear_model.LinearRegression(fit_intercept=False)
rerg.fit(Xbar,y)
print(w)
print(rerg.coef_)