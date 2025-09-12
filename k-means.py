from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
means = [[2,2],[8,5],[5,4]]
cov = [[1,0],[0,1]]
size = 500
X0 = np.random.multivariate_normal(means[0],cov,size)
X1 = np.random.multivariate_normal(means[1],cov,size)
X2 = np.random.multivariate_normal(means[2],cov,size)
# X3 = [[1,1],[2,2]]
# X4 = [[1,1],[2,3]]
# X5 = [[1,1],[2,5]] test
X = np.concatenate((X0,X1,X2),axis = 0)
original_label = np.asarray([0] * size + [1] * size + [2] * size).T
K = 3
# ham de in cac diem phan phoi chuan quanh center
def kmeans_display(X,label):
    K = np.amax(label) + 1
    X0 = X[label == 0,:]
    X1 = X[label == 1,:]
    X2 = X[label == 2,:]
    
    plt.plot(X0[:,0],X0[:,1],'b^',markersize = 4,alpha = .8)
    plt.plot(X1[:,0],X1[:,1],'go',markersize = 4,alpha = .8)
    plt.plot(X2[:,0],X2[:,1],'rs',markersize = 4,alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
def kmeans_init_centers(X,k):
    return X[np.random.choice(X.shape[0],k,replace=False)]
a = [[0,1],
     [1,2],
     [0,3]]
b = [[0,2],[0,5]]
print(cdist(a,b))
D = cdist(a,b)
print(np.argmin(D,axis = 1))
def kmeans_assign_labels(X,centers):
    D = cdist(X,centers)
    return np.argmin(D,axis = 1)
def kmeans_update_center(X,label,K):
    center = np.zeros((K,X.shape[1]))

    for k in range(K):
        Xk = X[label == k,:]

        center[k,:] = np.mean(Xk,axis = 0)
    return center
def has_converged(centers,new_centers):
    return (set(tuple(a) for a in centers) 
            == set(tuple(a) for a in new_centers)
            )
# def kmeans(X,K):
#     Xr = kmeans_init_centers(X,K)
#     i = 0
#     while(i < 5):
#         original_label = kmeans_assign_labels(X,Xr)
#         Xr = kmeans_update_center(X,original_label,K)
#         i = i + 1
#     return Xr
def kmeans(X,K):
    label = []
    Xr = [kmeans_init_centers(X,K)]
    i = 0
    while(i < 5):
        label.append(kmeans_assign_labels(X,Xr[-1]))
        new_center = kmeans_update_center(X,label[-1],K)
        if(has_converged(Xr[-1],new_center) == True):
            break
        Xr.append(new_center)
        i = i + 1
    return Xr[-1]
print(kmeans(X,K))
kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
print(kmeans.cluster_centers_)
kmeans_display(X,original_label)
predict_label = kmeans.predict(X)
kmeans_display(X,predict_label)
