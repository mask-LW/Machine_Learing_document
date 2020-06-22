import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import scipy.optimize as opt
file1 = '/Users/mac/Documents/ML-homework-Octave/machine-learning-ex2/ex2/ex2data1.txt'
file2= '/Users/mac/Documents/ML-homework-Octave/machine-learning-ex2/ex2/ex2data2.txt'

#加载数据
data = pd.read_csv(file2, header=None, names=['Test 1', 'Test 2', 'Accepted'])
#用图表表示样本，positive代表被录取，negative表示没被录取，对应离散值分别为0，1
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='black', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='yellow', marker='x', label='Not Accepted')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
#plt.show()

#实现逻辑回归
#Compute Cost and Gradient

#处理数据，需要给x向量的前面添加一列全为1，先插入data再分离
data.insert(0,'ones',1)
#data 为100x4
cols = data.shape[1]
#print(data.shape)
X = data.iloc[:,0:cols-1];
y = data.iloc[:,cols-1:cols];

#转化为矩阵
X = np.mat(X.values)
y = np.mat(y.values)





#非线性决策边界不能用线性的直线来分开两种类型，
#一种可行方法就是用高次项的feature，以产生非线性的decision boundary。
# 此时要考虑过拟合的问题。因此我们加上正则项。

def sigmoid(z):
    return  1/(1+np.exp(-z))

#返回新的feature
def feature_mapping(x, y, power, as_ndarray=False):
#     """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)

def costFunctionReg(theta, X, y, learningRate):

    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    print(theta.shape)  # 28x1
    print(X.shape)  # 118x28
    print(y.shape)  # 118x1
    h = sigmoid(np.dot(X, theta))
    first = np.multiply(-y, np.log(h))
    second = np.multiply((1 - y), np.log(1 - h))
    J = np.sum(first - second)/ len(X)
    print(J)
    theta_temp = theta.T
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta_temp[:,1:theta_temp.shape[1]], 2))
    print(reg)
    return J + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    print(theta.shape)  # 28x1
    print(X.shape)  # 118x28
    print(y.shape)  # 118x1
    h = sigmoid(np.dot(X, theta))
    grad = np.zeros((theta.shape[0],theta.shape[1]))
    #print('grad.shape' + str(grad.shape))
    error = h - y
    #print('error.shape' + str(error.shape))
    #print('theta.shape' + str(theta.shape))

    for i in range(theta.shape[0]):
        #print(X[:, i])
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[i,:])


    return grad

def costFunctionReg1(theta, X, y, learningRate):

    theta = np.mat(theta).T
    X = np.mat(X)
    y = np.mat(y)

    h = sigmoid(np.dot(X, theta))
    first = np.multiply(-y, np.log(h))
    second = np.multiply((1 - y), np.log(1 - h))
    J = np.sum(first - second)/ len(X)

    theta_temp = theta.T
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta_temp[:,1:theta_temp.shape[1]], 2))

    return J + reg


def gradientReg1(theta, X, y, learningRate):
    theta = np.mat(theta).T
    X = np.mat(X)
    y = np.mat(y)

    h = sigmoid(np.dot(X, theta))
    grad = np.zeros((theta.shape[0],theta.shape[1]))
    #print('grad.shape' + str(grad.shape))
    error = h - y
    #print('error.shape' + str(error.shape))
    #print('theta.shape' + str(theta.shape))

    for i in range(theta.shape[0]):
        #print(X[:, i])
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[i,:])


    return grad

def predict(theta,X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

x1 = data['Test 1']
x2 = data['Test 2']
X = np.mat(feature_mapping(x1,x2,6))


initial_theta = np.zeros((X.shape[1],1))

lambda_temp = 1
cost = costFunctionReg(initial_theta,X,y,lambda_temp)
print('Cost at initial theta (zeros): '+ str(cost))
print('Expected cost (approx): 0.693\n')
grad = gradientReg(initial_theta, X, y,lambda_temp)
print('Gradient at test theta - first five values only:\n')
for i in range(5):
    print(np.round(grad[i],4))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

test_theta = np.ones((X.shape[1],1))

lambda_temp = 10
cost = costFunctionReg(test_theta,X,y,lambda_temp)
print('Cost at initial theta (zeros)(with lambda = 10): '+ str(cost))
print('Expected cost (approx): 3.16\n')
grad = gradientReg(test_theta, X, y,lambda_temp)
for i in range(5):
    print(np.round(grad[i],4))
print('Gradient at test theta - first five values only:\n')
#print(grad)
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

#使用优化算法
lambda_temp = 1
result = opt.fmin_tnc(func=costFunctionReg1, x0=initial_theta, fprime=gradientReg1, args=(X, y, lambda_temp))
print(result)

theta = np.mat(result[0]).T
theta_min = np.mat(result[0])
predictions = predict(theta_min, X)
print(classification_report(y, predictions))
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(map(int, correct)) / len(correct)*100
print ('accuracy = {:.2f}%'.format(accuracy))




def plotDecisionBoundary():
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')

    x = np.linspace(-1, 1.5, 250)
    xx, yy = np.meshgrid(x, x)

    z = np.mat(feature_mapping(xx.ravel(), yy.ravel(), 6))
    z = z @ result[0]
    z = z.reshape(xx.shape)

    plt.contour(xx, yy, z, 0)
    plt.ylim(-.8, 1.2)
    plt.show()

plotDecisionBoundary()