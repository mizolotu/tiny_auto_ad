import os
import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf
import autokeras as ak
import keras_tuner

from preprocess_data import load_dataset, split_data
from config import *
from autokeras.utils import utils
from sklearn.metrics import roc_auc_score

class ReshapeBlock(ak.Block):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.Reshape(self.shape)
        output_node = layer(input_node)
        return output_node

class DistanceBlock(ak.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        input_node1 = tf.nest.flatten(inputs)[0]
        input_node2 = tf.cast(tf.nest.flatten(inputs)[1], tf.float32)
        layer = tf.keras.layers.Dense(
            hp.Int("num_units", min_value=input_node2.shape[1], max_value=input_node2.shape[1], step=32)
        )
        output_node = layer(input_node1)
        output_node = tf.expand_dims(tf.math.reduce_sum(tf.math.square(tf.math.subtract(output_node, input_node2)), axis=-1), -1)
        return output_node

class NoWeightsRegressionHead(ak.RegressionHead):

    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = tf.clip_by_value(inputs[0], 0, 1)
        output_node = tf.keras.layers.Lambda(lambda x: x, name=self.name)(input_node)
        print(output_node)
        return output_node

class EarlyStoppingAtMaxAuc(tf.keras.callbacks.Callback):

    def __init__(self, patience=10, max_fpr=1.0):
        super(EarlyStoppingAtMaxAuc, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.current = -np.Inf
        self.max_fpr = max_fpr

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if np.greater(self.current, self.best):
            self.best = self.current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_test_end(self, logs=None):
        probs = []
        testy = []
        for x, y in self.validation_data:
            y_labels = y[:, -1]
            predictions = self.model.predict(x)
            new_probs = predictions.flatten()
            probs = np.hstack([probs, new_probs])
            testy = np.hstack([testy, y_labels])
        self.current = roc_auc_score(testy, probs, max_fpr=self.max_fpr)
        print(f'\nValidation AUC:', self.current)

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Autokeras unsupervised.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    parser.add_argument('-t', '--trials', help='Number of trials', default=1, type=int)
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['raw', 'pam'])
    args = parser.parse_args()

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=32, series_step=4, labels=labels, feature_extractors=args.feature_extractors)
    data = split_data(target_dataset, train_on_anomalies=False, validate_on_anomalies=True, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]

    n_tr = data['tr'][0].shape[0]
    x_tr = np.reshape(data['tr'][0], newshape=(n_tr, np.prod(inp_shape)))
    n_val = data['val'][0].shape[0]
    x_val = np.reshape(data['val'][0], newshape=(n_val, np.prod(inp_shape)))
    n_inf = data['inf'][0].shape[0]
    x_inf = np.reshape(data['inf'][0], newshape=(n_inf, np.prod(inp_shape)))

    print(n_tr, n_val, n_inf, inp_shape)

    input_node1 = ak.StructuredDataInput()
    input_node2 = ak.StructuredDataInput()
    output_node = ak.Normalization()(input_node1)
    if len(inp_shape) > 1:
        output_node = ReshapeBlock(shape=inp_shape)(output_node)
        output_node = ak.ConvBlock()(output_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = DistanceBlock()([output_node, input_node2])
    output_node = NoWeightsRegressionHead(loss="mean_squared_error", metrics=[tf.keras.metrics.AUC(from_logits=True)])(output_node)

    project_name = '_'.join(['automodel', args.dataset, *[str(fe) for fe in args.feature_extractors]])

    clf = ak.AutoModel(
        project_name=project_name,
        inputs=[input_node1, input_node2],
        outputs=output_node,
        #objective='val_binary_accuracy',
        objective=keras_tuner.Objective("val_auc", direction="max"),
        overwrite=True,
        max_trials=args.trials
    )
    clf.fit(
        [x_tr, x_tr], data['tr'][1],
        validation_data=([x_val, x_val], data['val'][1]),
        epochs=1000,
        batch_size=512,
        callbacks=[
            #tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=100, mode='max', restore_best_weights=True)
            EarlyStoppingAtMaxAuc(patience=100)
        ],
        verbose=2
    )

    model = clf.export_model()
    model.summary()

    model_fpath = osp.join('unsupervised_model_autokeras', args.dataset, *[str(fe) for fe in args.feature_extractors])
    #model.save(model_fpath)

    #loaded_model = tf.keras.models.load_model(model_fpath, custom_objects=ak.CUSTOM_OBJECTS)
    p = clf.predict([x_inf, x_inf])
    auc = roc_auc_score(data['inf'][1], p)
    print(auc)
    h = clf.evaluate([x_inf, x_inf], data['inf'][1])
    print(h)