import numpy as np
import argparse as arp
import os.path as osp

from matplotlib import pyplot as pp
from sklearn.manifold import TSNE
from preprocess_data import load_dataset, split_data
from config import *


class CentroidClusteringAnomalyDetector:

    def __init__(self):
        self.trained = False

    def _calculate_distances(self, data, eps=1e-10):
        E_va_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = np.min(D_va, axis=1)
        self.radiuses = np.zeros((C_.shape[0], 2))
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0

    def predict(self, data, alpha, eps=1e-10):
        if self.trained:
            radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
            E_te_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
            C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
            D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
            cl_labels_te = np.argmin(D_te, axis=1)
            dists_te = np.min(D_te, axis=1)
            nte = E_te_.shape[0]
            pred_thrs = radiuses[cl_labels_te]
            predictions = np.zeros(nte)
            predictions[np.where(dists_te > pred_thrs)[0]] = 1
        else:
            predictions = None
        return predictions

    def evaluate(self, data, alpha_range=np.arange(0, 10, 0.01), fpr_max=1.0):

        alpha_best, fpr_, tpr_ = None, None, None
        acc_max = 0

        for alpha in alpha_range:
            predictions = self.predict(data, alpha)
            if predictions is not None:
                acc = len(np.where(predictions == data[1])[0]) / data[1].shape[0]
                fpr = len(np.where((predictions == 1) & (data[1] == 0))[0]) / (1e-10 + len(np.where(data[1] == 0)[0]))
                tpr = len(np.where((predictions == 1) & (data[1] == 1))[0]) / (1e-10 + len(np.where(data[1] == 1)[0]))

                #print(acc, alpha, len(np.where(predictions == 0)[0]), len(np.where(predictions == 1)[0]))
                if acc > acc_max and fpr <= fpr_max:
                    acc_max = acc
                    alpha_best = alpha
                    fpr_ = fpr
                    tpr_ = tpr
            else:
                acc_max, fpr_, tpr_ = 0, 0, 0
                alpha_best = None
                break

        return acc_max, alpha_best, fpr_, tpr_

    def tsne_plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, eps=1e-10):
        if self.trained:
            nc = self.centroids.shape[0]
            X_plot = np.vstack([
                (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
                (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            ])
            tsne = TSNE(n_components=2, learning_rate='auto', init='random')
            X_tsne = tsne.fit_transform(X_plot)
            pp.style.use('default')
            pp.figure(figsize=(12, 7))
            scatter_centroids = pp.scatter(X_tsne[:nc, 0], X_tsne[:nc, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 2000, alpha=0.5);
            scatter_points = pp.scatter(X_tsne[nc:, 0], X_tsne[nc:, 1], c=data[1], s=10, cmap='Accent', marker='x')
            pp.xlabel('t-SNE feature 1', fontsize=10)
            pp.ylabel('t-SNE feature 2', fontsize=10)
            pp.legend(
                scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
                labels + ['cluster centroids'],
                loc=0
            )
            fname = f'{prefix}_' if prefix is not None else ''
            fname += f'tsne_{self.__class__.__name__}_{nc}.pdf'
            pp.savefig(osp.join(fig_dir, fname))
            pp.close()
        else:
            print('You are trying to plot untrained classifier!')


class ScalableKmeans(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(ScalableKmeans).__init__()

    def fit(self, data, data_rad, n_clusters, batch_size=16, l=4, n_iters=100):

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

        self._calculate_distances(data_rad)

        self.trained = True


class ClustreamKmeans(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(ScalableKmeans).__init__()

    def fit(self, data, data_rad, n_clusters, n_micro_clusters=16, micro_cluster_radius_alpha=3, n_iters=100, eps=1e-10):

        ntr = data[0].shape[0]
        n_features = data[0].shape[1]

        # init min and max values, centroids, and their weights

        self.xmin = np.inf * np.ones(n_features)
        self.xmax = -np.inf * np.ones(n_features)

        C, R = [], []

        # the main clustering loop

        for xi in range(ntr):

            # take a sample

            x = data[0][xi, :].copy()

            # update min and max values

            self.xmin = np.min(np.vstack([self.xmin, x]), 0)
            self.xmax = np.max(np.vstack([self.xmax, x]), 0)

            # create initial micro-clsuter

            if len(C) < 2:
                C.append([1, x, x ** 2])
                R.append(0)

            # add sample to the existing framework

            else:

                # update the minimal distance between micro-clusters

                D = np.zeros((len(C), len(C)))
                for i in range(len(C)):
                    for j in range(len(C)):
                        if i < j:
                            D[i, j] = np.sqrt(np.sum(((C[i][1] / C[i][0] - self.xmin) / (self.xmax - self.xmin + eps) - (C[j][1] / C[j][0] - self.xmin) / (self.xmax - self.xmin + eps)) ** 2))
                        else:
                            D[i, j] = np.inf
                cl_dist_min = np.min(D)
                i_dmin, j_dmin = np.where(D == cl_dist_min)
                i_dmin = i_dmin[0]
                j_dmin = j_dmin[0]

                # update micro-cluster radiuses

                for i in range(len(C)):
                    if C[i][0] < 2:
                        R[i] = cl_dist_min
                    else:
                        ls_ = (C[i][1] - C[i][0] * self.xmin) / (self.xmax - self.xmin + eps)
                        ss_ = (C[i][2] - 2 * C[i][1] * self.xmin + C[i][0] * self.xmin ** 2) / ((self.xmax - self.xmin) ** 2 + eps)
                        R[i] = micro_cluster_radius_alpha * np.mean(np.sqrt(np.clip(ss_ / C[i][0] - (ls_ / C[i][0]) ** 2, 0, np.inf)))
                        if R[i] == 0:
                            R[i] = cl_dist_min

                # calculate distances from the sample to the micro-clusters

                D = np.zeros(len(C))
                for i in range(len(C)):
                    D[i] = np.sqrt(np.sum(((x - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[i][1] / C[i][0] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2))
                k = np.argmin(D)

                if D[k] <= R[k]:

                    # add sample to the existing micro-cluster

                    C[k][0] += 1
                    C[k][1] += x
                    C[k][2] += x ** 2

                else:

                    # merge the closest clusters

                    if len(C) == n_micro_clusters:
                        C[i_dmin][0] += C[j_dmin][0]
                        C[i_dmin][1] += C[j_dmin][1]
                        C[i_dmin][2] += C[j_dmin][2]
                        C[j_dmin][0] = 1
                        C[j_dmin][1] = x
                        C[j_dmin][2] = x ** 2
                        # print(f'Micro-clusters {i_dmin} and {j_dmin} have been merged')

                    # create a new cluster

                    else:
                        C.append([1, x, x ** 2])
                        R.append(0)

        C = np.vstack([c[1] / c[0] for c in C])

        # weighted k-means clustering

        count = np.hstack([c[0] for c in C])
        centroids = C[np.random.choice(range(C.shape[0]), n_clusters, replace=False), :]

        for i in range(n_iters):

            D = np.zeros((C.shape[0], centroids.shape[0]))
            for j in range(C.shape[0]):
                for k in range(centroids.shape[0]):
                    D[j, k] = np.sum(((C[j, :] - self.xmin) / (self.xmax - self.xmin + eps) - (centroids[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            cl_labels = np.argmin(D, axis=1)

            centroids_new = []
            W_new = []

            for j in range(n_clusters):
                idx = np.where(cl_labels == j)[0]
                if len(idx) > 0:
                    centroids_new.append(np.sum(count[idx, None] * C[idx, :], axis=0) / (np.sum(count[idx] + eps)))
                    W_new.append(np.sum(count[idx]))
                else:
                    pass

            if np.array_equal(centroids, centroids_new):
                break

            centroids = np.vstack(centroids_new)
            self.weights = np.hstack(W_new)

        self.centroids = np.array(centroids)

        self._calculate_distances(data_rad)

        self.trained = True


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
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    parser.add_argument('-i', '--methods', help='Method index', type=int, default=[0, 1], nargs='+', choices=[i for i in range(len(methods))])
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=32, series_step=4, labels=labels, feature_extractors=['fft', 'pam'])
    data = split_data(target_dataset, shuffle_features=False)

    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_tries = 10
    fpr_max = 0.0

    for i in args.methods:
        method = locals()[methods[i]]
        m = method()
        acc_method = 0
        for n_clusters in cluster_range:
            acc_max, acc_sum, fpr_sum, tpr_sum = 0, 0, 0, 0
            alpha_best = None
            for j in range(n_tries):
                m.fit(data['tr'], n_clusters=n_clusters, data_rad=data['val'])
                acc, alpha, fpr, tpr = m.evaluate(data['inf'], fpr_max=fpr_max)
                acc_sum += acc
                fpr_sum += fpr
                tpr_sum += tpr
                if acc > acc_max:
                    acc_max = acc
                    alpha_best = alpha
            if acc_sum > acc_method:
                acc_method = acc_sum
                m.tsne_plot(data['inf'], prefix=dataset)
            print(f'{m.__class__.__name__} with {n_clusters} clusters and hyperparameter {alpha}: acc = {acc_sum / n_tries}, fpr = {fpr_sum / n_tries}, tpr = {tpr_sum / n_tries}')