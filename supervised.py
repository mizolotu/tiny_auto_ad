import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from preprocess_data import load_dataset, split_data
from config import *

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'fan':
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset == 'bearing':
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=128, labels=labels, feature_extractor='simple_features')
    data = split_data(target_dataset, train_on_normal=True, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]
    xmin = np.min(data['tr'][0], 0)
    xmax = np.max(data['tr'][0], 0)

    inputs = tf.keras.layers.Input(shape=inp_shape)
    hidden = (inputs - xmin) / (xmax - xmin + 1e-10)
    if len(inp_shape) == 2:
        hidden = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=1, activation='relu')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(units=512, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Dense(units=512, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4), metrics='binary_accuracy')
    model.summary()
    model.fit(
        *data['tr'],
        validation_data=data['val'],
        epochs=10000,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
        ]
    )
    h = model.evaluate(*data['inf'])
    print(h)
