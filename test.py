import numpy as np

from sklearn.cluster import AffinityPropagation

def calculate_r(a, s, k):
    return s[k] - np.max(np.hstack([s[:k], s[k+1:]]) + np.hstack([a[:k], a[k+1:]]))

def calculate_a(r, k, i):
    vec = np.maximum(np.hstack([r[:i], r[i+1:]]), 0)
    if k == i:
        return np.sum(vec)
    else:
        return min(0, r[k] + np.sum(vec))

def update_R(A, S, R):
    for i in range(X.shape[0]):
        for k in range(X.shape[0]):
            R[i,k] = calculate_r(A[i, :], S[i, :], k)
    return R

def update_A(R, A):
    for i in range(X.shape[0]):
        for k in range(X.shape[0]):
            A[i, k] = calculate_a(R[:,k], k, i)
    return A

if __name__ =='__main__':

    X = np.array([
        [3, 4, 3, 2, 1],
        [4, 3, 5, 1, 1],
        [3, 5, 3, 3, 3],
        [2, 1, 3, 3, 2],
        [1, 1, 3, 2, 3],
    ])

    af = AffinityPropagation()
    af_ = af.fit(X)

    print(af_.cluster_centers_)

    D = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

    S = -D
    S[np.arange(len(X)), np.arange(len(X))] = np.min(S)

    A = np.zeros((X.shape[0], X.shape[0]))
    R = np.zeros((X.shape[0], X.shape[0]))

    n = 100
    lmbd = 0.5

    for i in range(n):

        R_ = R.copy()
        A_ = A.copy()

        for i in range(X.shape[0]):
            for k in range(X.shape[0]):
                R[i, k] = S[i, k] - np.max(np.hstack([S[i, :k], S[i, k+1:]]) + np.hstack([A[i, :k], A[i, k+1:]]))

        A = update_A(R, A)

        if np.all(A == A_) and np.all(R == R_):
            break

        if i != 0:
            R = lmbd * R_ + (1 - lmbd) * R
            A = lmbd * A_ + (1 - lmbd) * A

    exemplar_ids = np.argmax(A + R, 1)

    print(exemplar_ids)
