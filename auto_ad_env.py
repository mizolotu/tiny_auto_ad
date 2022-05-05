import numpy as np
import gym, os, json
import os.path as osp

from gym import spaces
from sklearn.metrics import auc

class AutoAdEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, datasets, n_steps, n_features,
                 n_samples=1000, n_clusters_max=8, a_max=7, n_x=10, n_unif=10, n_em_points=10, n_mv_points=10):
        super(AutoAdEnv, self).__init__()

        self.algorithms = [
            '_scalable_kmeans',
            '_clustream_kmeans'
        ]

        self.datasets = datasets
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_algorithms = len(self.algorithms)
        self.n_clusters_range = np.arange(2, n_clusters_max + 1)
        self.n_a_range = np.arange(1, a_max + 1)
        self.n_em_points = n_em_points
        self.n_mv_points = n_mv_points
        self.n_steps = n_steps

        self.volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        self.x_unif = np.random.uniform(np.zeros(n_features), np.ones(n_features), size=(n_unif, n_features))

        obs_dim = n_features * 2 + self.n_algorithms + len(self.n_clusters_range) + len(self.n_a_range) + n_em_points + n_mv_points
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=float)
        self.action_space = spaces.MultiDiscrete(nvec=[self.n_algorithms, len(self.n_clusters_range), len(self.n_a_range)])

    def step(self, action):

        #alg = self.algorithms[action[0]]
        alg = self.algorithms[0]
        k = self.n_clusters_range[action[1]]
        a = self.n_clusters_range[action[2]]

        data = self._sample_data()

        xmin, xmax, centroids, weights = getattr(AutoAdEnv, alg)(self, data['tr'], k)

        radiuses = self._calculate_radiuses(data['val'], xmin, xmax, centroids, a)

        reward = self._calculae_reward(data['inf'], xmin, xmax, centroids, radiuses)
        self.rewards[self.step_idx] = reward

        x = (data['tr'][0] - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        x_alg = np.zeros(len(self.algorithms))
        x_alg[action[0]] = 1
        x_ncl = np.zeros(len(self.n_clusters_range))
        x_ncl[action[1]] = 1
        x_aaa = np.zeros(len(self.n_a_range))
        x_aaa[action[2]] = 1

        mv_points = np.zeros(self.n_mv_points)
        em_points = np.zeros(self.n_em_points)

        observation = np.hstack([np.mean(x, axis=0), np.std(x, axis=0), x_alg, x_ncl, x_aaa, mv_points, em_points])

        info = {}
        if self.step_idx >= self.n_steps - 1:
            done = True
            info['episode'] = {
                "r": np.max(self.rewards),
                "l": self.step_idx,
            }
        else:
            done = False

        self.step_idx += 1

        return observation, reward, done, info

    def reset(self):
        self.dataset = np.random.choice(self.datasets)
        self.step_idx = 0
        self.rewards = np.zeros(self.n_steps)
        observation = np.zeros(self.observation_space.shape)
        return observation

    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def _sample_data(self, inf_split=0.3, val_split=0.3):

        idx0 = np.where(self.dataset['Y'] == 0)[0]
        np.random.shuffle(idx0)
        idx0 = idx0[:self.n_samples]
        split = {}
        split['inf'], train_val = np.split(idx0, [int(inf_split * len(idx0))])
        split['val'], split['tr'] = np.split(train_val, [int(val_split * len(train_val))])

        idx1 = np.where(self.dataset['Y'] == 1)[0]
        idx1 = np.random.choice(idx1, len(split['inf']), replace=True)
        split['inf'] = np.append(split['inf'], idx1)

        idx = np.arange(self.dataset['X'].shape[1])
        np.random.shuffle(idx)
        idx = idx[:self.n_features]

        X = self.dataset['X'][:, idx]
        Y = self.dataset['Y'].squeeze()
        data = {}
        for key in split.keys():
            data[key] = [X[split[key], :], Y[split[key]]]

        return data

    def _scalable_kmeans(self, data, n_clusters, batch_size=16, l=4):

        # init min and max values, centroids, and their weights

        xmin = np.inf * np.ones(self.n_features)
        xmax = -np.inf * np.ones(self.n_features)
        C, W = None, None

        # the main clustering loop

        ntr = data[1].shape[0]

        for i in range(0, ntr - batch_size, batch_size):

            # take a batch

            idx = np.arange(i, i + batch_size)
            B = data[0][idx, :]

            # update min and max values

            xmin = np.min(np.vstack([xmin, B]), 0)
            xmax = np.max(np.vstack([xmax, B]), 0)

            # pick initial centroids

            if C is None:
                C = B[np.random.choice(range(B.shape[0]), n_clusters, replace=False), :]
                D = np.zeros((B.shape[0], C.shape[0]))
                for j in range(B.shape[0]):
                    for k in range(C.shape[0]):
                        D[j, k] = np.sum(((B[j, :] - xmin) / (xmax - xmin + 1e-10) - (C[k, :] - xmin) / (xmax - xmin + 1e-10)) ** 2)
                min_dist = np.zeros(D.shape)
                min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
                count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
                W = np.zeros(n_clusters)

            # select candidates

            D = np.zeros((B.shape[0], C.shape[0]))
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    D[j, k] = np.sum(((B[j, :] - xmin) / (xmax - xmin + 1e-10) - (C[k, :] - xmin) / (xmax - xmin + 1e-10)) ** 2)
            cost = np.sum(np.min(D, axis=1))
            p = np.min(D, axis=1) / (cost + 1e-10)
            C = np.r_[C, B[np.random.choice(range(len(p)), l, p=(p + 1e-10) / np.sum(p + 1e-10), replace=False), :]]

            # assign data to the centroids

            D = np.zeros((B.shape[0], C.shape[0]))
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    D[j, k] = np.sum(((B[j, :] - xmin) / (xmax - xmin + 1e-10) - (C[k, :] - xmin) / (xmax - xmin + 1e-10)) ** 2)
            min_dist = np.zeros(D.shape)
            min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
            count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
            for i in range(len(W)):
                count[i] += W[i]

            # weighted k-means clustering

            n_iters = 100
            centroids = C[:n_clusters, :]
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

        return  xmin, xmax, C, W

    def _clustream_kmeans(self, k, a):

        return  xmin, xmax, centroids, d_mean, d_std

    def _calculate_radiuses(self, data, xmin, xmax, centroids, alpha):

        E_va_ = (data[0] - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        C_ = (centroids - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)

        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = np.min(D_va, axis=1)

        radiuses = np.zeros(C_.shape[0])
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                radiuses[k] = np.mean(dists_va[idx]) + alpha * np.std(dists_va[idx])
            else:
                radiuses[k] = 0

        return radiuses

    def _calculae_reward(self, data, xmin, xmax, centroids, radiuses):

        E_te_ = (data[0] - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        C_ = (centroids - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_te = np.argmin(D_te, axis=1)
        dists_te = np.min(D_te, axis=1)

        nte = E_te_.shape[0]

        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1

        return len(np.where(predictions == data[1])[0]) / nte * 100


    def _em(self, volume_support, s_unif, s_X, n_generated, t_max=0.999, t_step=0.1):
        t = np.arange(0, 1 / volume_support, t_step / volume_support)
        EM_t = np.zeros(t.shape[0])
        n_samples = s_X.shape[0]
        s_X_unique = np.unique(s_X)
        EM_t[0] = 1.
        for u in s_X_unique:
            EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() - t * (s_unif > u).sum() / n_generated * volume_support)
        amax = np.argmax(EM_t <= t_max) + 1
        if amax == 1:
            amax = -1
        AUC = auc(t[:amax], EM_t[:amax])
        return AUC, EM_t, amax

    def _mv(self, volume_support, s_unif, s_X, n_generated, alpha_step, alpha_min=0.9, alpha_max=0.999):
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
            mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
        return auc(axis_alpha, mv), mv