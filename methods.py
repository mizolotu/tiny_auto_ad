import numpy as np
import argparse as arp

from preprocess_data import load_dataset, split_data

from config import *


class ScalableKmeans:

    def __init__(self):
        pass

    def fit(self, data, n_clusters, batch_size=16, l=4, n_iters=100):

        n_features = data[0].shape[1]

        # init min and max values, centroids, and their weights

        self.xmin = np.inf * np.ones(n_features)
        self.xmax = -np.inf * np.ones(n_features)

        C, W = None, None

        # the main clustering loop

        ntr = data[1].shape[0]

        for i in range(0, ntr - batch_size, batch_size):

            # take a batch

            idx = np.arange(i, i + batch_size)
            B = data[0][idx, :]

            # update min and max values

            self.xmin = np.min(np.vstack([self.xmin, B]), 0)
            self.xmax = np.max(np.vstack([self.xmax, B]), 0)

            # pick initial centroids

            if C is None:
                C = B[np.random.choice(range(B.shape[0]), n_clusters, replace=False), :]
                D = np.zeros((B.shape[0], C.shape[0]))
                for j in range(B.shape[0]):
                    for k in range(C.shape[0]):
                        D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
                min_dist = np.zeros(D.shape)
                min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
                count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
                W = np.zeros(n_clusters)

            # select candidates

            D = np.zeros((B.shape[0], C.shape[0]))
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            cost = np.sum(np.min(D, axis=1))
            p = np.min(D, axis=1) / (cost + 1e-10)
            C = np.r_[C, B[np.random.choice(range(len(p)), l, p=(p + 1e-10) / np.sum(p + 1e-10), replace=False), :]]

            # assign data to the centroids

            D = np.zeros((B.shape[0], C.shape[0]))
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            min_dist = np.zeros(D.shape)
            min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
            count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
            for i in range(len(W)):
                count[i] += W[i]

            # weighted k-means clustering

            centroids = C[:n_clusters, :]

            for i in range(n_iters):

                D = np.zeros((C.shape[0], centroids.shape[0]))
                for j in range(C.shape[0]):
                    for k in range(centroids.shape[0]):
                        D[j, k] = np.sum(((C[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (centroids[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
                cl_labels = np.argmin(D, axis=1)

                centroids_new = []
                W_new = []

                for j in range(n_clusters):
                    idx = np.where(cl_labels == j)[0]
                    if len(idx) > 0:
                        centroids_new.append(np.sum(count[idx, None] * C[idx, :], axis=0) / (np.sum(count[idx] + 1e-10)))
                        W_new.append(np.sum(count[idx]))
                    else:
                        pass

                if np.array_equal(centroids, centroids_new):
                    break

                centroids = np.vstack(centroids_new)
                self.weights = np.hstack(W_new)

            self.centroids = np.array(centroids)

    def validate(self):
        

if __name__ == '__main__':

    methods = [
        'ScalableKmeans',
        'ClustreamKmeans',
        'Strap',
        'WeightedCmeans',
        'UbiquitousSom',
        'GngOnline',
        'GrowWhenRequired',
        'IncrementalGng',
        'Gstream'
    ]

    parser = arp.ArgumentParser(description='Test AD methods.')
    parser.add_argument('-i', '--index', help='Method index', type=int, default=0, choices=[i for i in range(len(methods))])
    args = parser.parse_args()

    method = locals()[methods[args.index]]

    target_dataset = load_dataset(TE_DATA_DIR, series_len=32, labels={0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']})

    data = split_data(target_dataset)

    m = method()
    m.fit(data['tr'], n_clusters=5)

    print(m.centroids, m.weights, m.xmin, m.xmax)