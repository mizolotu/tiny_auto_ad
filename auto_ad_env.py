import numpy as np
import gym

from gym import spaces
from sklearn.metrics import auc

class AutoAdEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, datasets, n_steps, n_features, n_samples=5000, n_clusters_max=8, a_max=9, n_unif=20, n_em_points=20, n_mv_points=20):
        super(AutoAdEnv, self).__init__()

        self.algorithms = [
            '_scalable_kmeans',
            '_clustream_kmeans'
        ]

        self.datasets = datasets
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_algorithms = len(self.algorithms)
        self.n_clusters = [2, n_clusters_max]
        self.std_multiplier = [1, a_max]
        self.n_em_points = n_em_points
        self.n_mv_points = n_mv_points
        self.n_steps = n_steps

        self.volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        self.x_unif = np.random.uniform(np.zeros(n_features), np.ones(n_features), size=(n_unif, n_features))

        act_dim = 3
        obs_dim = n_features * 2 + n_em_points + n_mv_points + act_dim
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=float)
        self.action_space = spaces.Box(shape=(act_dim,), low=-1, high=1)

    def step(self, action):

        #print(f'Step {self.step_idx + 1}')

        alg_idx = int(np.round((action[0] + 1) / 2 * (self.n_algorithms - 1)))
        alg = self.algorithms[alg_idx]

        k = int(np.round((action[1] + 1) / 2 * (self.n_clusters[1] - self.n_clusters[0]))) + self.n_clusters[0]
        a = (action[2] + 1) / 2 * (self.std_multiplier[1] - self.std_multiplier[0]) + self.std_multiplier[0]

        data = self._sample_data()

        xmin, xmax, centroids, weights = getattr(AutoAdEnv, alg)(self, data['tr'], k)

        radiuses = self._calculate_radiuses(data['val'], xmin, xmax, centroids, a)

        reward, s_x = self._calculate_reward(data['inf'], xmin, xmax, centroids, radiuses)
        self.rewards[self.step_idx] = reward

        x = (data['tr'][0] - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)

        _, s_unif = self._calculate_reward([self.x_unif, ], xmin, xmax, centroids, radiuses)

        _, em_points, _ = self._em(self.volume_support, s_unif, np.random.choice(s_x, len(s_unif)), len(s_unif), t_step=1.0/self.n_em_points)
        _, mv_points = self._mv(self.volume_support, s_unif, np.random.choice(s_x, len(s_unif)), len(s_unif), alpha_step=1.0/self.n_mv_points)

        observation = np.hstack([np.mean(x, axis=0), np.std(x, axis=0), action, mv_points, em_points])

        info = {}
        if self.step_idx >= self.n_steps - 1:
            done = True
        else:
            done = False

        info['episode'] = {
            "r": reward, # np.max(self.rewards),
            "l": self.step_idx + 1,
        }

        self.step_idx += 1

        return observation, reward, done, info

    def reset(self):
        dataset_idx = np.random.choice(np.arange(len(self.datasets)))
        self.dataset = self.datasets[dataset_idx]
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

    def _scalable_kmeans(self, data, n_clusters, batch_size=16, l=4, n_iters=100):

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

    def _clustream_kmeans(self, data, n_clusters, n_micro_clusters=16, micro_cluster_radius_alpha=3, n_iters=100):

        xmin = np.inf * np.ones(self.n_features)
        xmax = -np.inf * np.ones(self.n_features)
        C, R = [], []

        # the main clustering loop

        ntr = data[1].shape[0]
        for xi in range(ntr):

            # take a sample

            x = data[0][xi, :].copy()

            # update min and max values

            xmin = np.min(np.vstack([xmin, x]), 0)
            xmax = np.max(np.vstack([xmax, x]), 0)

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

        return  xmin, xmax, C, W

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

    def _calculate_reward(self, data, xmin, xmax, centroids, radiuses):

        E_te_ = (data[0] - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        C_ = (centroids - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
        D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_te = np.argmin(D_te, axis=1)
        dists_te = np.min(D_te, axis=1)

        nte = E_te_.shape[0]

        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + 1e-10)

        if len(data) > 1:
            acc = len(np.where(predictions == data[1])[0]) / nte
        else:
            acc = None

        return acc, scores

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

    def _mv(self, volume_support, s_unif, s_X, n_generated, alpha_step=0.1, alpha_min=0.9, alpha_max=0.999):
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