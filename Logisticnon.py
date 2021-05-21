import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loaddata():
    path = 'exm2.txt'
    data = pd.read_csv(path, names=['Test1', 'Test2', 'Accepted'], sep=',')

    return data


def feature_mapping(x1, x2, power):
    data = {}

    for i in np.arange(power+1):
        for j in np.arange(i+1):
            data['F{}{}'.format(i-j, j)] = np.power(x1, i-j)*np.power(x2, j)

    print(data)
    return pd.DataFrame(data)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def costFuction(X, y, theta, lr):
    A = sigmoid(X@theta)

    fist = y*np.log(A)
    sec = (1-y)*np.log(1-A)

    reg = np.sum(np.power(theta[1:], 2))*(lr/(2*len(X)))
    # 正则项
    return -np.sum(fist+sec)/len(X)+reg


def grandDesecent(X, y, theta, alpha, iters, lr):


    costs = []
    for i in range(iters):
        A = sigmoid(X @ theta)
        reg = theta[1:]*(lr/len(X))
        reg = np.insert(reg, 0, values=0, axis=0)
        theta = theta - (X.T@(A-y))*(alpha/len(X)) - reg
        cost = costFuction(X, y, theta, lr)
        costs.append(cost)

    return theta, costs


def predict(X, theta):
    prob =sigmoid(X@theta)

    return [1 if x >= 0.5 else 0 for x in prob]


def draw(theta_final):
    x = np.linspace(-1.2,1.2,200)
    xx,yy = np.meshgrid(x,x)
    z = feature_mapping(xx.ravel(),yy.ravel(),6).values
    zz = z@theta_final
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.scatter(data[data['Accepted'] == 0]['Test1'], data[data['Accepted'] == 0]['Test2'], c='r', marker='x', label='y=0')
    ax.scatter(data[data['Accepted'] == 1]['Test1'], data[data['Accepted'] == 1]['Test2'], c='b', marker='o', label='y=1')
    ax.legend()
    ax.set(xlabel='Test1', ylabel='Test2')
    plt.contour(xx, yy, zz, 0)
    plt.show()


if __name__ == '__main__':
    data = loaddata()
    x1 = data['Test1']
    x2 = data['Test2']
    data2 = feature_mapping(x1, x2, 6)
    X = data2.values
    y = data.iloc[:, -1].values
    y = y.reshape(len(y), 1)
    alpha = 0.001
    iters = 200000
    lr = 0.001
    theta = np.random.rand(28, 1)
    theta_final, costs = grandDesecent(X, y, theta, alpha, iters, lr)
    draw(theta_final)
    y_ = np.array(predict(X, theta_final))
    y_pre = y_.reshape(len(y), 1)
    print(np.mean(y_pre == y))


