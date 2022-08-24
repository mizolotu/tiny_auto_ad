import numpy as np

def relu(x):
    return np.array([item if item >= 0 else 0 for item in x])

def d_relu(x):
    return np.array([1 if item > 0 else 0 for item in x])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def forward(w0, b0, w1, b1, x):
    y0 = relu(np.dot(x, w0) + b0)
    y1 = sigmoid(np.dot(y0, w1) + b1)
    return y0, y1

def backward(w0, b0, w1, b1, x, y0, y1, y, lr=0.05):

    de1 = y - y1  # 1x2
    de0 = np.dot(de1, w1.T)  # 1x4

    dw0 = np.zeros((3, 4))
    for i in range(3):
        for j in range(4):
            dw0[i, j] = de0[j] * d_relu(y0)[j] * x[i]

    db0 = np.zeros(4)
    for j in range(4):
        db0[j] = de0[j] * d_relu(y0)[j]

    dw1 = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            dw1[i, j] = de1[j] * d_sigmoid(y1)[j] * y0[i]

    db1 = np.zeros(2)
    for j in range(2):
        db1[j] = de1[j] * d_sigmoid(y1)[j]

    return w0 + lr * dw0, b0 + lr * db0, w1 + lr * dw1, b1 + lr * db1

if __name__ == '__main__':

    X = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    w0 = np.ones((3, 4)) * 0.1
    #w0 = np.random.rand(3, 4)
    b0 = np.ones(4) * 0.2
    #b0 = np.random.rand(4)
    w1 = np.ones((4, 2)) * 0.1
    #w1 = np.random.rand(4, 2)
    b1 = np.ones(2) * 0.2
    #b1 = np.random.rand(2)


    Y = np.array([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]
    ])

    for iter in range(500):
        loss = 0
        for x, y in zip(X, Y):
            y0, y1 = forward(w0, b0, w1, b1, x)
            loss += 0.5 * np.sum((y1 - y) ** 2)
            w0, b0, w1, b1 = backward(w0, b0, w1, b1, x, y0, y1, y)
        loss /= 2
        print(iter, loss)

    for x, y in zip(X, Y):
        y0, y1 = forward(w0, b0, w1, b1, x)
        print(y ,y1)

    print(w0, b0)
    print(w1, b1)


