import os
import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from preprocess_data import load_dataset, split_data
from config import *
from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib import pyplot as pp


class BGN(tf.keras.models.Model):

    def __init__(self, nfeatures, latent_dim, layers):
        super(BGN, self).__init__()
        self.nfeatures = nfeatures
        self.latent_dim = tf.constant(latent_dim)
        self.layers_ = layers

        # generator

        self.generator_layers = []
        for nunits in layers:
            self.generator_layers.append(tf.keras.layers.Dense(nunits))

        # discriminator

        self.discriminator_layers = []
        for nunits in layers:
            self.discriminator_layers.append(tf.keras.layers.Dense(nunits))
        self.discriminator_layers.append(tf.keras.layers.Flatten())
        self.discriminator_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

        # loss trackers

        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')

        self.built = False

    def build(self, input_shape):

        # generator

        self.generator_trainable_variables = []
        self.generator_layers[0].build(input_shape)
        for i in range(len(self.layers_) - 1):
            self.generator_layers[i + 1].build(input_shape=(None, self.layers_[i]))
        for i in range(len(self.generator_layers)):
            self.generator_trainable_variables.extend(self.generator_layers[i].trainable_variables)

        # discriminator

        self.discriminator_trainable_variables = []
        self.discriminator_layers[0].build((None, self.nfeatures))
        for i in range(len(self.layers_)):
            self.discriminator_layers[i + 1].build(input_shape=(None, self.layers_[i]))
        for i in range(len(self.discriminator_layers)):
            self.discriminator_trainable_variables.extend(self.discriminator_layers[i].trainable_variables)
        self.built = True

    def call(self, x):
        for layer in self.discriminator_layers:
            x = layer(x)
        score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(x), logits=x)
        return score[:, 0]

    def train_step(self, data):
        x_real, z_with_label = data
        z, _ = tf.split(z_with_label, [self.latent_dim, 1], axis=1)
        z = tf.expand_dims(z, 1)
        x_fake = z
        for layer in self.generator_layers:
            x_fake = layer(x_fake)
        d_preds = tf.concat([x_fake, x_real], axis=0)
        for layer in self.discriminator_layers:
            d_preds = layer(d_preds)
        pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)
        d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + tf.reduce_mean(tf.nn.softplus(-pred_e))
        g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))
        d_gradients = tf.gradients(d_loss, self.discriminator_trainable_variables)
        g_gradients = tf.gradients(g_loss, self.generator_trainable_variables)
        self.optimizer.apply_gradients(zip(d_gradients, self.discriminator_trainable_variables))
        self.optimizer.apply_gradients(zip(g_gradients, self.generator_trainable_variables))
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
        }

        return d_loss, g_loss

    def test_step(self, data):
        x_real, z_with_label = data
        z, _ = tf.split(z_with_label, [self.latent_dim, 1], axis=1)
        z = tf.expand_dims(z, 1)
        x_fake = z
        for layer in self.generator_layers:
            x_fake = layer(x_fake)
        d_preds = tf.concat([x_fake, x_real], axis=0)
        for layer in self.discriminator_layers:
            d_preds = layer(d_preds)
        pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + tf.reduce_mean(tf.nn.softplus(-pred_e))
        g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
        }

        return d_loss, g_loss


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

    #tr_data_std = (data['tr'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)
    #val_data_std = (data['val'][0] - np.min(data['tr'][0], 0)[None, :]) / (np.max(data['tr'][0], 0)[None, :] - np.min(data['tr'][0], 0)[None, :] + 1e-10)
    #tr_data_std = (data['tr'][0] - np.mean(data['tr'][0], 0)[None, :]) / (np.std(data['tr'][0], 0)[None, :] + 1e-10)
    #val_data_std = (data['val'][0] - np.mean(data['tr'][0], 0)[None, :]) / (np.std(data['tr'][0], 0)[None, :] + 1e-10)
    tr_data_std = data['tr'][0]
    val_data_std = data['val'][0]

    model = BGN(inp_shape[0], 3, [64, 32])
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
    pp.plot(fpr, tpr)
    pp.xlabel('FPR')
    pp.ylabel('TPR')
    pp.savefig('som_roc.png')
    pp.close()


