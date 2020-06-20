import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import scipy.optimize as opt
file1 = '/Users/mac/Documents/ML-homework-Octave/machine-learning-ex2/ex2/ex2data1.txt'

#加载数据
data = pd.read_csv(file1, header=None, names=['Exam 1', 'Exam 2', 'IsAdmitted'])
#用图表表示样本，positive代表被录取，negative表示没被录取，对应离散值分别为0，1
# positive = data[data['IsAdmitted'].isin([1])]
# negative = data[data['IsAdmitted'].isin([0])]
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='black', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='yellow', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

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
X = np.matrix(X.values)
y = np.matrix(y.values)


initial_theta = np.zeros((X.shape[1],1))


#sigmoid function
def sigmoid(z):
    return  1/(1+np.exp(-z))




def costfuncion(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    m = len(X)
    J = np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1-h)) )/m
    return J

#print(np.dot(X,theta)) == print(X * theta)

def gradient(theta,X,y):
    m = len(X)
    theta = theta.reshape(X.shape[1],1)
    grad = np.zeros((theta.shape[0], theta.shape[1]))
    h = sigmoid(X*theta)
    errors = h-y
    grad = X.T * errors/m
    return grad

def costFunction(theta, X, y):
    #由优化算法传入来的参数发生变化，需要对其重新处理
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = sigmoid(np.dot(X, theta.T))
    m = len(X)
    J = np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1-h)) )/m
    return J

#print(np.dot(X,theta)) == print(X * theta)

def gradientDescent(theta,X,y):
    m = len(X)
    theta = theta.reshape(X.shape[1],1)
    grad = np.zeros((theta.shape[0], theta.shape[1]))
    h = sigmoid(X*theta)
    errors = h-y
    grad = X.T * errors/m
    return grad

def predict(theta, X):
    probability = sigmoid(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]

#参数为0时
cost = costfuncion(initial_theta,X,y)
print('Cost at initial theta (zeros):' + str(cost) + ' (This value should be about 0.693)')
grad = gradient(initial_theta, X, y)
print('Gradient at initial theta (zeros):' + 'Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')
print(grad)
print('***********************************')
#改变参数测试：
test_theta = np.mat([[-24, 0.2, 0.2]]).reshape(3,1)
cost = costfuncion(test_theta,X,y)
print('Cost at initial theta (zeros):' + str(cost) + ' (This value should be about 0.218)')
grad = gradient(test_theta, X, y)
print('Gradient at initial theta (zeros):' + 'Expected gradients (approx):\n 0.043\n 2.566\n 2.647')
print(grad)
print('***********************************')


#Optimizing using fminunc

result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=gradientDescent, args=(X, y))
print('result:',str(result))
print('cost : ' + str(costFunction(result[0], X, y)) + ' (This value should be about 0.203)')
theta = np.mat(result[0]).T
print(theta)

#Predict and Accuracies
ex = np.mat([[1 , 45 , 85]])
prob = sigmoid(ex*theta)
print("For a student with scores 45 and 85, we predict an admission :" + str(prob))

#Compute accuracy on our training set
predictions = predict(theta, X)

#print("train accuracy: {} %".format(100 - np.mean(p==y) * 100))

print(classification_report(y, predictions))
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

#画出决策边界

