import numpy as np
from mnist import MNIST # require python-mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from scipy.spatial.distance import cdist
mndata = MNIST(r'C:\Users\Administrator\Downloads\MNIST_ORG')
# f = open(r'C:\Users\Administrator\Downloads\MNIST_ORG\t10k-labels.idx1-ubyte')
# mndata.load_testing()
# x = mndata.test_images

# lis = [0]*60000
# for i in range(60000):
#     lis[i] = i
# center = random.sample(lis,10)
center_label = [8881, 3749, 55121, 10399, 50344, 16169, 10580, 40792, 59132, 35931]
center = x[0][center_label]
label = [0,1,2,3,4,5,6,7,8,9]
print(center)
def change_label(x,center):
    a = cdist(center,x[0][:])
    b = np.zeros((1,60000))
    for i in range(60000):
        b[0][i] = np.argmin(a,axis=1)
    return b
def update_center(x):
    center = np.zeros((10,256))
    for i in range(10):
        XK = x[0][x[1][:] == i]
        center[i][:] = np.mean(XK,axis = 0)
    return center
def kmeans(x,center):
    i = 0
    a = center
    while i < 20 :
        x[1][:] = change_label(x,a)
        a = update_center(x)
        i = i + 1
    return x,a
a,center = kmeans(x,center)
print(a[1][0])
print(x[1][0])
