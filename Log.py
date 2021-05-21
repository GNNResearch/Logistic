import matplotlib.pyplot as plt
import numpy as np


def Dataset():
    X_ = []
    y_ = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        X_.append([1, float(lineArr[0]), float(lineArr[1])])
        y_.append(int(lineArr[2]))
        fr.close()
    X = np.array(X_)
    y = np.array(y_).reshape(len(y_), 1)

    return X, y


def sigmiod(z):
    return 1/(1+np.exp(-z))


def costFuction(X, y, theta):
    A = sigmiod(X@theta)
    first = y*np.log(A)
    sec = (1-y)*np.log(1-A)

    return -np.sum(first+sec)/len(X)


def gradientDesent(X, y, theta, alpha, iters):

    costs = []

    for i in range(iters):
        A = sigmiod(X @ theta)
        theta = theta - (X.T@(A-y))*(alpha/len(X))
        cost = costFuction(X, y, theta)
        costs.append(cost)

    return theta


def predict(X, theta):
    prob = sigmiod(X@theta)

    return [1 if x >= 0.5 else 0 for x in prob]


def draw(X, y, theta_final):
    coef1 = -theta_final[0, 0]/theta_final[2, 0]
    coef2 = -theta_final[1, 0]/theta_final[2, 0]

    t = np.linspace(-3, 3, 15)
    m = coef1+coef2*t

    n = np.shape(y)[0]
    flg,ax = plt.subplots()
    for i in range(n):
        if int(y[i] == 1):
            ax.scatter(X[i][1], X[i][2], c='r', marker='o')
        else:
            ax.scatter(X[i][1], X[i][2], c='b', marker='o')
    ax.set(xlabel='X1',ylabel='X2')
    ax.plot(t, m, c='g')
    plt.show()


if __name__ == '__main__':
    X, y = Dataset()
    theta = np.random.rand(3, 1)
    alpha = 0.1
    iters = 1000
    theta_final = gradientDesent(X, y, theta, alpha, iters)
    print(theta_final)
    draw(X, y, theta_final)
    y_= np.array(predict(X, theta_final)).reshape(len(X), 1)

    print(np.mean(y_ == y))
