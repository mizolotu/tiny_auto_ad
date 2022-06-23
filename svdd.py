import os
import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from preprocess_data import load_dataset, split_data
from config import *
from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib import pyplot as pp

class Svdd(tf.keras.models.Model):

    def __init__(self, preprocessor, nu=0.05):
        super(Svdd, self).__init__()
        self.nu = nu
        self.preprocessor = preprocessor
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.c = tf.reduce_mean(self.preprocessor(X), 0)
        self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=False)
        self.built = True

    def call(self, x):
        x = self.preprocessor(x)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        return scores

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            x = self.preprocessor(inputs)
            dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
            scores = dists - self.R ** 2
            penalty = tf.maximum(scores, tf.zeros_like(scores))
            loss = self.R ** 2 + (1 / self.nu) * penalty
            #loss = tf.reduce_mean(dists)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        test = tf.sort(tf.math.sqrt(dists))[tf.cast((1 - self.nu) * tf.math.reduce_sum(tf.ones_like(dists)), tf.int32)]
        self.R.assign(test)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 2:
            inputs, outputs = data
        else:
            inputs, outputs = data[0]
        x = self.preprocessor(inputs)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        penalty = tf.maximum(scores, tf.zeros_like(scores))
        loss = self.R ** 2 + (1 / self.nu) * penalty
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['pam'])
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

    tr_data_std = data['tr'][0] # (data['tr'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)
    val_data_std = data['val'][0] # (data['val'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)

    inputs = tf.keras.layers.Input(shape=inp_shape)
    hidden = (inputs - np.mean(tr_data_std, 0)[None, :]) / (np.std(tr_data_std, 0)[None, :] + 1e-10)
    #hidden = inputs
    hidden = tf.keras.layers.Dense(units=64)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.ReLU()(hidden)
    hidden = tf.keras.layers.Dense(units=32)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.ReLU()(hidden)
    encoded = tf.keras.layers.Dense(units=16)(hidden)
    encoder = tf.keras.models.Model(inputs, encoded)

    hidden = tf.keras.layers.Dense(units=32)(encoded)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.ReLU()(hidden)
    hidden = tf.keras.layers.Dense(units=64)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.ReLU()(hidden)
    outputs = tf.keras.layers.Dense(units=inp_shape[0])(hidden)

    autoencoder = tf.keras.models.Model(inputs, outputs)
    autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    autoencoder.fit(
        tr_data_std, tr_data_std,
        validation_data=(val_data_std, val_data_std),
        epochs=10000,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
        ]
    )

    for e_layer, a_layer in zip(encoder.weights, autoencoder.weights):
        e_layer.assign(a_layer)

    model = Svdd(preprocessor=encoder, nu=0.01)
    model.build(input_shape=(None, *inp_shape), X=tr_data_std)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    model.fit(
        tr_data_std, data['tr'][1],
        validation_data=(val_data_std, data['val'][1]),
        epochs=10000,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
        ]
    )

    model.summary()

    p = np.clip(model.predict(data['val'][0]), 0, np.inf)
    alpha = 5
    thr = np.mean(p) + alpha * np.std(p)
    predictions = np.zeros(len(data['inf'][1]))
    y_pred = np.clip(model.predict(data['inf'][0]), 0, 1)
    predictions[np.where(y_pred > thr)[0]] = 1
    acc = len(np.where(predictions == data['inf'][1])[0]) / data['inf'][1].shape[0]
    fpr = len(np.where((predictions == 1) & (data['inf'][1] == 0))[0]) / (1e-10 + len(np.where(data['inf'][1] == 0)[0]))
    tpr = len(np.where((predictions == 1) & (data['inf'][1] == 1))[0]) / (1e-10 + len(np.where(data['inf'][1] == 1)[0]))
    auc = roc_auc_score(data['inf'][1], y_pred)
    print(f'Accuracy = {acc}, TPR = {tpr}, FPR = {fpr}, AUC = {auc}')
    fpr, tpr, thresholds = roc_curve(data['inf'][1], y_pred)
    pp.plot(fpr, tpr, '.')
    pp.savefig('tmp.png')
    pp.close()


