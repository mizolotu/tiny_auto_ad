import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from preprocess_data import load_dataset, split_data
from config import *
from sklearn.metrics import auc, roc_auc_score, roc_curve

class Svdd(tf.keras.models.Model):

    def __init__(self, preprocessor, nu):
        super(Svdd, self).__init__()
        self.nu = nu
        self.preprocessor = preprocessor
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.c = tf.reduce_mean(self.preprocessor(X), 0)
        self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R')
        self.built = True

    def call(self, x):
        x = self.preprocessor(x)
        return tf.reduce_sum(tf.square(x - self.c), axis=-1)

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            x = self.preprocessor(inputs)
            y_pred = tf.reduce_sum(tf.square(x - self.c), axis=-1)
            loss = tf.reduce_mean(y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 2:
            inputs, outputs = data
        else:
            inputs, outputs = data[0]
        x = self.preprocessor(inputs)
        y_pred = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        loss = tf.reduce_mean(y_pred)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['fft', 'pam'])
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=32, series_step=8, labels=labels, feature_extractors=args.feature_extractors)
    data = split_data(target_dataset, train_on_anomalies=False, validate_on_anomalies=False, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]

    model_fpath = osp.join('model_autokeras', args.dataset, *[str(fe) for fe in args.feature_extractors])

    inputs = tf.keras.layers.Input(shape=inp_shape)
    hidden = (inputs - np.mean(data['tr'][0], 0)[None, :]) / (np.std(data['tr'][0], 0)[None, :] + 1e-10)
    hidden = tf.keras.layers.Dense(units=512)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.ReLU()(hidden)
    hidden = tf.keras.layers.Dense(units=512)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    outputs = tf.keras.layers.ReLU()(hidden)
    preprocessor = tf.keras.models.Model(inputs, outputs)

    model = Svdd(preprocessor=preprocessor, nu=0.1)
    model.build(input_shape=(None, *inp_shape), X=data['tr'][0])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    model.fit(
        *data['tr'],
        validation_data=data['val'],
        epochs=10000,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
        ]
    )

    model.summary()

    p = np.clip(model.predict(data['inf'][0]), 0, 1)
    alpha = 3
    thr = np.mean(p) + alpha * np.std(p)
    predictions = np.zeros(len(p))
    predictions[np.where(predictions > thr)[0]] = 1
    acc = len(np.where(predictions == data['inf'][1])[0]) / data['inf'][1].shape[0]
    fpr = len(np.where((predictions == 1) & (data['inf'][1] == 0))[0]) / (1e-10 + len(np.where(data['inf'][1] == 0)[0]))
    tpr = len(np.where((predictions == 1) & (data['inf'][1] == 1))[0]) / (1e-10 + len(np.where(data['inf'][1] == 1)[0]))
    auc = roc_auc_score(data['inf'][1], p)
    print(f'Accuracy = {acc}, TPR = {tpr}, FPR = {fpr}, AUC = {auc}')


