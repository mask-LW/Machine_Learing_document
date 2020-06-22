import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y

X, y = load_data('ex3data1.mat',transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
print('process X:'+str(X.shape))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']
theta1, theta2 = load_weight('ex3weights.mat')
print('theta1 :'+str(theta1.shape))
print('theta2 :'+str(theta2.shape))


print('-------')
a0 = X
z1 = np.dot(a0, theta1.T) # (5000, 401) @ (25,401).T = (5000, 25)
print(z1.shape)
z1 = np.insert(z1, 0, values=np.ones(z1.shape[0]), axis=1)
a1 = sigmoid(z1)
print(a1.shape)
z2 = np.dot(a1, theta2.T)
print(z2.shape)
a2 = sigmoid(z2)
print(a2.shape)

# numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
y_pred = np.argmax(a2, axis=1) + 1
print(y_pred.shape)
print(y.shape)
print(classification_report(y, y_pred))