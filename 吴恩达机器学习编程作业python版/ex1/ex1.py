import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#文件路径
file1 = '/Users/mac/Documents/ML-homework-Octave/machine-learning-ex1/ex1/ex1data1.txt'
file12 = '/Users/mac/Documents/ML-homework-Octave/machine-learning-ex1/ex1/ex1data2.txt'

#单一变量的线性回归问题

#导入文件处理为两个变量,分离符为','
data = pd.read_csv(file1, header=None, names=['Population', 'Profit'])

#绘图展示数据
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
#plt.show();

#定义代价函数
def computeCost(X, y ,theta):
    predictions = X * theta.T;
    sqrtErrores = np.power((predictions - y), 2);
    return np.sum(sqrtErrores) / (2 * len(X))
#需要给x向量的前面添加一列全为1，先插入data再分离
data.insert(0,'ones',1);

cols = data.shape[1];
#print(cols)
X = data.iloc[:,0:cols-1];
#print(X);

y = data.iloc[:,cols-1:cols];
#print(y);

#使用矩阵运算
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0.0,0.0])
#print(y);
J = computeCost(X,y,theta);
print('test computeCost')
print('参数为0时，其代价函数的值为：')
print(J)

#初始化参数
alpha=0.01;
iterations=1500;

def gradientDescent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
    print(parameters)
    cost = np.zeros(iterations)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m

    for i in range(iterations):
        # 利用向量化一步求解
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        cost[i] = computeCost(X, y, theta)
        theta = temp
    return theta,cost;


final_theta, cost = gradientDescent(X, y, theta, alpha, iterations);


print(final_theta)


#绘制图像来查看模型拟合数据的情况
#x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
#f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润
#fig, ax = plt.subplots(figsize=(6,4))
#ax.plot(x, f, 'r', label='Prediction')
#ax.scatter(data['Population'], data.Profit, label='Traning Data')
#ax.legend(loc=2)  # 2表示在左上角
#ax.set_xlabel('Population')
#ax.set_ylabel('Profit')
#ax.set_title('Predicted Profit vs. Population Size')
#plt.show()