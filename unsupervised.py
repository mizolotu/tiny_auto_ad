import numpy as np
import argparse as arp
import os.path as osp

from matplotlib import pyplot as pp
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_auc_score, roc_curve
from preprocess_data import load_dataset, split_data
from config import *


class CentroidClusteringAnomalyDetector:

    def __init__(self):
        self.trained = False

    def _em(self, volume_support, s_U, s_X, n_generated, t_max=0.999, t_step=0.001):
        t = np.arange(0, 1 / volume_support, t_step / volume_support)
        EM_t = np.zeros(t.shape[0])
        n_samples = s_X.shape[0]
        s_X_unique = np.unique(s_X)
        EM_t[0] = 1.
        for u in s_X_unique:
            EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() - t * (s_U > u).sum() / n_generated * volume_support)
        amax = np.argmax(EM_t <= t_max) + 1
        if amax == 1:
            amax = -1
        AUC = auc(t[:amax], EM_t[:amax])
        return AUC, EM_t, amax

    def _mv(self, volume_support, s_U, s_X, n_generated, alpha_step=0.0001, alpha_min=0.99, alpha_max=0.9999):
        axis_alpha = np.arange(alpha_min, alpha_max, alpha_step * (alpha_max - alpha_min))
        n_samples = s_X.shape[0]
        s_X_argsort = s_X.argsort()
        mass = 0
        cpt = 0
        u = s_X[s_X_argsort[-1]]
        mv = np.zeros(axis_alpha.shape[0])
        for i in range(axis_alpha.shape[0]):
            while mass < axis_alpha[i]:
                cpt += 1
                u = s_X[s_X_argsort[-cpt]]
                mass = 1. / n_samples * cpt
            mv[i] = float((s_U >= u).sum()) / n_generated * volume_support
        return auc(axis_alpha, mv), mv

    def _calculate_distances(self, data, eps=1e-10):
        E_va_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = np.min(D_va, axis=1)
        self.radiuses = np.zeros((C_.shape[0], 3))
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def _set_radiuses(self, data, metric='em', alpha_max=10, alpha_step=0.1, n_generated=100000):

        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(np.zeros(n_features), np.ones(n_features), size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha_min = np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10))
        print(alpha_min)
        alpha_range = np.arange(alpha_min, alpha_max, alpha_step)
        metric_vals = np.zeros(len(alpha_range))
        for i, alpha in enumerate(alpha_range):
            _, s_X = self.predict(data, alpha)
            if s_X is not None:
                _, s_U = self.predict(X_unif, alpha, standardize=False)
                metric_vals[i] = metric_fun(volume_support, s_U, s_X, n_generated)[0]
        if metric == 'em':
            alpha = alpha_range[np.argmax(metric_vals)]
            metric_val = np.max(metric_vals)
        elif metric == 'mv':
            alpha = alpha_range[np.argmin(metric_vals)]
            metric_val = np.min(metric_vals)
        return alpha, metric_val

    def predict(self, data, alpha, eps=1e-10, standardize=True):
        if self.trained:
            radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
            if standardize:
                E_te_ = (data - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
            else:
                E_te_ = data
            C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
            D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
            cl_labels_te = np.argmin(D_te, axis=1)
            dists_te = np.min(D_te, axis=1)
            nte = E_te_.shape[0]
            pred_thrs = radiuses[cl_labels_te]
            predictions = np.zeros(nte)
            predictions[np.where(dists_te > pred_thrs)[0]] = 1
            scores = (pred_thrs - dists_te) / (pred_thrs + 1e-10)
        else:
            predictions, scores = None, None
        return predictions, scores

    def evaluate(self, data, alpha):
        predictions, _ = self.predict(data[0], alpha)
        if predictions is not None:
            acc = len(np.where(predictions == data[1])[0]) / data[1].shape[0]
            fpr = len(np.where((predictions == 1) & (data[1] == 0))[0]) / (1e-10 + len(np.where(data[1] == 0)[0]))
            tpr = len(np.where((predictions == 1) & (data[1] == 1))[0]) / (1e-10 + len(np.where(data[1] == 1)[0]))
        else:
            acc, fpr, tpr = 0, 0, 0

        return acc, fpr, tpr

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
        super(ScalableKmeans, self).__init__()

    def fit(self, data, data_rad, n_clusters, batch_size=16, l=4, n_iters=100, metric='em'):

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

        alpha, metric_val = self._set_radiuses(data[0], metric=metric)

        self.trained = True

        return alpha, metric_val


class ClustreamKmeans(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(ClustreamKmeans, self).__init__()

    def fit(self, data, data_rad, n_clusters, n_micro_clusters=16, micro_cluster_radius_alpha=3, n_iters=100, eps=1e-10, metric='em'):

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

        alpha, metric_val = self._set_radiuses(data[0], metric=metric)

        self.trained = True

        return alpha, metric_val


class WeightedAffinityPropagation(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(WeightedAffinityPropagation, self).__init__()

    def fit(self):

        ntr = data[0].shape[0]
        n_features = data[0].shape[1]

        self.xmin = np.inf * np.ones(n_features)
        self.xmax = -np.inf * np.ones(n_features)
        C, R = [], []

        # the main clustering loop

        ntr = data[1].shape[0]
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
                            D[i, j] = np.sqrt(np.sum(((C[i][1] / C[i][0] - xmin) / (xmax - xmin + 1e-10) - (C[j][1] / C[j][0] - xmin) / (xmax - xmin + 1e-10)) ** 2))
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
                        ls_ = (C[i][1] - C[i][0] * xmin) / (xmax - xmin + 1e-10)
                        ss_ = (C[i][2] - 2 * C[i][1] * xmin + C[i][0] * xmin ** 2) / ((xmax - xmin) ** 2 + 1e-10)
                        R[i] = micro_cluster_radius_alpha * np.mean(np.sqrt(np.clip(ss_ / C[i][0] - (ls_ / C[i][0]) ** 2, 0, np.inf)))
                        if R[i] == 0:
                            R[i] = cl_dist_min

                # calculate distances from the sample to the micro-clusters

                D = np.zeros(len(C))
                for i in range(len(C)):
                    D[i] = np.sqrt(np.sum(((x - xmin) / (xmax - xmin + 1e-10) - (C[i][1] / C[i][0] - xmin) / (xmax - xmin + 1e-10)) ** 2))
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

        W = np.hstack([c[0] for c in C])
        C = np.vstack([c[1] / c[0] for c in C])

        # weighted k-means clustering

        count = np.hstack([c[0] for c in C])
        centroids = C[np.random.choice(range(C.shape[0]), n_clusters, replace=False), :]

        for i in range(n_iters):

            D = np.zeros((C.shape[0], centroids.shape[0]))
            for j in range(C.shape[0]):
                for k in range(centroids.shape[0]):
                    D[j, k] = np.sum(((C[j, :] - xmin) / (xmax - xmin + 1e-10) - (centroids[k, :] - xmin) / (xmax - xmin + 1e-10)) ** 2)
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
            W = np.hstack(W_new)

        C = np.array(centroids)

        return xmin, xmax, C, W


if __name__ == '__main__':

    methods = [
        'ScalableKmeans',
        'ClustreamKmeans',
        'WeightedAffinityPropagation',
        'WeightedCmeans',
        'UbiquitousSom',
        'GngOnline',
        'GrowWhenRequired',
        'IncrementalGng',
        'Gstream'
    ]

    parser = arp.ArgumentParser(description='Test AD methods.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='fan', choices=['fan', 'bearing'])
    parser.add_argument('-i', '--methods', help='Method index', type=int, default=[0, 1], nargs='+', choices=[i for i in range(len(methods))])
    parser.add_argument('-t', '--tries', help='Number of tries', default=1, type=int)
    parser.add_argument('-c', '--clusters', help='Cluster range', default=[2, 3, 4, 5, 6, 7, 8, 9, 10], type=int, nargs='+')
    parser.add_argument('-m', '--metric', help='Metric', default='em', choices=['em', 'mv'])
    parser.add_argument('-s', '--seed', help='Seed', default=0, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=32, series_step=32, labels=labels, feature_extractors=['pam'])
    data = split_data(target_dataset, shuffle_features=False)

    cluster_range = args.clusters
    n_tries = args.tries

    for i in args.methods:
        method = locals()[methods[i]]
        m = method()
        acc_method = 0
        for n_clusters in cluster_range:
            acc_sum, fpr_sum, tpr_sum, metric_sum = 0, 0, 0, 0
            for j in range(n_tries):
                alpha, metric_val = m.fit(data['tr'], n_clusters=n_clusters, data_rad=data['val'], metric=args.metric)
                acc, fpr, tpr = m.evaluate(data['inf'], alpha)
                acc_sum += acc
                fpr_sum += fpr
                tpr_sum += tpr
                metric_sum += metric_val
            #if acc_sum > acc_method:
            #    acc_method = acc_sum
            #    m.tsne_plot(data['inf'], prefix=dataset)
            print(f'{m.__class__.__name__} with {n_clusters} clusters on average: acc = {acc_sum / n_tries}, fpr = {fpr_sum / n_tries}, tpr = {tpr_sum / n_tries}, {args.metric} = {metric_sum / n_tries}')