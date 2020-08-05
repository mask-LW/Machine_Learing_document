import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据
def load_data():
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


X, y, Xval, yval, Xtest, ytest = load_data()

# df = pd.DataFrame({'water_level': X, 'flow': y})
# sns.lmplot('water_level', 'flow', data=df, fit_reg=False, height=7)
# plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

#Regularized Linear Regression Cost
def cost(theta, X, y):
    m = X.shape[0]
    temp = np.dot(X,theta) - y
    J = np.dot(temp.T,temp)/2/m
    return J
def  regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta[1:]
    regularized_term = (l / 2 / m) * np.dot(regularized_term.T,regularized_term)
    return regularized_term + cost(theta, X, y)

theta = np.ones(X.shape[1])
# J = regularized_cost(theta, X, y,1)
# print('Cost at theta = [1 ; 1]: ' , J)

def gradient(theta, X, y):
    m = X.shape[0]
    temp = np.dot(X, theta) - y
    grad = np.dot(X.T,temp)/m
    return grad

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term

# grad = regularized_gradient(theta, X, y, l=1)
# print('Gradient at theta = [1 ; 1]: ' , grad)

#Train linear regression with lambda = 0


def linear_regression( X , y , l):
    theta = np.ones(X.shape[1])
    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

# theta = np.ones(X.shape[1])
# final_theta = linear_regression(theta,X, y, l=0).get('x')
# b = final_theta[0] # intercept
# m = final_theta[1] # slope
# print('intercept:',b)
# print('slope',m)
# plt.scatter(X[:,1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()

#Learning Curve for Linear Regression
# training_cost, cv_cost = [], []
# m = X.shape[0]
# for i in range(1, m + 1):
#     #     print('i={}'.format(i))
#     res = linear_regression(theta,X[:i, :], y[:i], l=0)
#
#     tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
#     cv = regularized_cost(res.x, Xval, yval, l=0)
#     #     print('tc={}, cv={}'.format(tc, cv))
#
#     training_cost.append(tc)
#     cv_cost.append(cv)

# plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
# plt.legend(loc=1)
# plt.show()

#Feature Mapping for Polynomial Regression
# X, y, Xval, yval, Xtest, ytest = load_data()
#
# def poly_features(X, power):
#     X_poly = np.zeros([X.shape[0] , power])
#     for i in range(0, power):
#         X_poly[:, i] = np.power(X,i+1)
#
#     return X_poly
#
# def get_mean_std(X):
#     means = np.mean(X, axis=0)  # 按列求均值 (1,power+1)
#     stds = np.std(X, axis=0)  # (1,7)
#     return means, stds
#
#
# def feature_normalize(X, means, stds):
#     X[:, 1:] = (X[:, 1:] - means[1:]) / stds[1:]
#     return X
#
# X_poly=poly_features(X,8)
# X_means, X_std = get_mean_std(X_poly)
# X_norm=feature_normalize(X_poly, X_means, X_std)
#
# Xval_poly=poly_features(Xval,8)
# Xval_means, Xval_std = get_mean_std(Xval_poly)
# Xval_norm=feature_normalize(Xval_poly, Xval_means, Xval_std)
#
# Xtest_poly=poly_features(Xtest,8)
# Xtest_means, Xtest_std = get_mean_std(Xtest_poly)
# Xtest_norm=feature_normalize(Xtest_poly, Xtest_means, Xtest_std)

X, y, Xval, yval, Xtest, ytest = load_data()

def poly_features(X, power):
    X_poly = np.zeros([X.shape[0] , power])
    for i in range(0, power):
        X_poly[:, i] = np.power(X,i+1)

    return X_poly

def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    m = df.shape[1]
    for i in range(0,m):

        mean = df[:,i].mean();
        std = df[:, i].std();
        df[:, i] = (df[:, i] - mean) / std

    return df

def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature

        df = poly_features(x, power=power)
        # normalization
        ndarr = normalize_feature(df)

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]

X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)
temp = X_poly[:3,:]
#print(temp)

def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)

# plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
# plt.show()

l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)

plt.xlabel('lambda')

plt.ylabel('cost')
#plt.show()

for l in l_candidate:
    theta = linear_regression(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))