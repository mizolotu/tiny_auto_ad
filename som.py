import os
import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from preprocess_data import load_dataset, split_data
from config import *
from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib import pyplot as pp

class SOMLayer(tf.keras.layers.Layer):

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        self.initial_prototypes = prototypes
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.prototypes = self.add_weight(shape=(self.nprototypes, *input_dims), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=-1)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.nprototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def som_loss(weights, distances):
    return tf.reduce_mean(tf.reduce_sum(weights * distances, axis=1))

class SOM(tf.keras.models.Model):

    def __init__(self, map_size, batchnorm, T_min=0.1, T_max=10.0, niterations=10000, nnn=4):
        super(SOM, self).__init__()
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        ranges = [np.arange(m) for m in map_size]
        mg = np.meshgrid(*ranges, indexing='ij')
        self.prototype_coordinates = tf.convert_to_tensor(np.array([item.flatten() for item in mg]).T)
        self.bn_layer = tf.keras.layers.BatchNormalization(trainable=batchnorm)
        self.som_layer = SOMLayer(map_size, name='som_layer')
        self.T_min = T_min
        self.T_max = T_max
        self.niterations = niterations
        self.current_iteration = 0
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.nnn = nnn

    @property
    def prototypes(self):
        return self.som_layer.get_weights()[0]

    def call(self, x):
        x = self.bn_layer(x)
        x = self.som_layer(x)
        print(x)
        s = tf.sort(x, axis=1)
        spl = tf.split(s, [self.nnn, self.nprototypes - self.nnn], axis=1)
        return tf.reduce_mean(spl[0], axis=1)

    def map_dist(self, y_pred):
        labels = tf.gather(self.prototype_coordinates, y_pred)
        mh = tf.reduce_sum(tf.math.abs(tf.expand_dims(labels, 1) - tf.expand_dims(self.prototype_coordinates, 0)), axis=-1)
        return tf.cast(mh, tf.float32)

    @staticmethod
    def neighborhood_function(d, T):
        return tf.math.exp(-(d ** 2) / (T ** 2))

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:

            # Compute cluster assignments for batches

            inputs = self.bn_layer(inputs)
            d = self.som_layer(inputs)
            y_pred = tf.math.argmin(d, axis=1)

            # Update temperature parameter

            self.current_iteration += 1
            if self.current_iteration > self.niterations:
                self.current_iteration = self.niterations
            self.T = self.T_max * (self.T_min / self.T_max) ** (self.current_iteration / (self.niterations - 1))

            # Compute topographic weights batches

            w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

            # calculate loss

            loss = som_loss(w_batch, d)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {
            "total_loss": self.total_loss_tracker.result()
        }

    def test_step(self, data):
        inputs, outputs = data
        inputs = self.bn_layer(inputs)
        d = self.som_layer(inputs)
        y_pred = tf.math.argmin(d, axis=1)
        w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)
        loss = som_loss(w_batch, d)
        self.total_loss_tracker.update_state(loss)
        return {
            "total_loss": self.total_loss_tracker.result()
        }

def som(nsteps, nfeatures, layers=[64, 64], dropout=0.5, batchnorm=True, lr=5e-5):
    model = SOM(layers, dropout, batchnorm)
    model.build(input_shape=(None, nsteps, nfeatures))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr))
    return model, 'som_{0}'.format('-'.join([str(item) for item in layers])), 'ad'

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['fft', 'pam'])
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    parser.add_argument('-s', '--seed', help='Seed', default=0, type=int)
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    args = parser.parse_args()

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=32, series_step=1, labels=labels, feature_extractors=args.feature_extractors)
    data = split_data(target_dataset, train_on_anomalies=False, validate_on_anomalies=False, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]

    tr_data_std = (data['tr'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)
    val_data_std = (data['val'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)
    tr_data_std = (data['tr'][0] - np.mean(data['tr'][0], 0)[None, :]) / (np.std(data['tr'][0], 0)[None, :] + 1e-10)
    val_data_std = (data['val'][0] - np.mean(data['tr'][0], 0)[None, :]) / (np.std(data['tr'][0], 0)[None, :] + 1e-10)
    #tr_data_std = data['tr'][0]
    #val_data_std = data['val'][0]

    model = SOM([64, 64], batchnorm=False)
    model.build(input_shape=(None, inp_shape[0]))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))

    model.fit(
        tr_data_std, data['tr'][1],
        validation_data=(val_data_std, data['val'][1]),
        epochs=10000,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=100, mode='min', restore_best_weights=True)
        ]
    )

    model.summary()
    p = np.clip(model.predict(data['val'][0]), 0, np.inf)
    alpha = 3
    thr = np.mean(p) + alpha * np.std(p)
    predictions = np.zeros(len(data['inf'][1]))
    y_pred = np.clip(model.predict(data['inf'][0]), 0, 1)
    print(y_pred, thr)
    predictions[np.where(y_pred > thr)[0]] = 1
    acc = len(np.where(predictions == data['inf'][1])[0]) / data['inf'][1].shape[0]
    fpr = len(np.where((predictions == 1) & (data['inf'][1] == 0))[0]) / (1e-10 + len(np.where(data['inf'][1] == 0)[0]))
    tpr = len(np.where((predictions == 1) & (data['inf'][1] == 1))[0]) / (1e-10 + len(np.where(data['inf'][1] == 1)[0]))
    auc = roc_auc_score(data['inf'][1], y_pred)
    print(f'Accuracy = {acc}, TPR = {tpr}, FPR = {fpr}, AUC = {auc}')
    fpr, tpr, thresholds = roc_curve(data['inf'][1], y_pred)
    pp.plot(fpr, tpr, 'o')
    pp.xlabel('FPR')
    pp.ylabel('TPR')
    pp.savefig('som_roc.png')
    pp.close()


