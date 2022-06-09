import os
import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf
import autokeras as ak

from preprocess_data import load_dataset, split_data
from config import *


class ReshapeBlock(ak.Block):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.Reshape(self.shape)
        output_node = layer(input_node)
        return output_node

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing', choices=['fan', 'bearing'])
    parser.add_argument('-t', '--trials', help='Number of trials', default=1, type=int)
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    parser.add_argument('-f', '--feature_extrators', help='Feature extractors', nargs='+')
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
    target_dataset = load_dataset(data_fpath, series_len=32, labels=labels, feature_extractors=args.feature_extrators)
    data = split_data(target_dataset, train_on_normal=True, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]
    print(inp_shape)

    n_tr = data['tr'][0].shape[0]
    x_tr = np.reshape(data['tr'][0], newshape=(n_tr, np.prod(inp_shape)))
    n_val = data['val'][0].shape[0]
    x_val = np.reshape(data['val'][0], newshape=(n_val, np.prod(inp_shape)))
    n_inf = data['inf'][0].shape[0]
    x_inf = np.reshape(data['inf'][0], newshape=(n_inf, np.prod(inp_shape)))

    xmin = np.min(x_tr, 0)
    xmax = np.max(x_tr, 0)

    input_node = ak.StructuredDataInput()
    output_node = ak.Normalization()(input_node)
    if len(inp_shape) > 1:
        output_node = ReshapeBlock(shape=inp_shape)(output_node)
        output_node = ak.ConvBlock()(output_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead(loss="binary_crossentropy", metrics=["binary_accuracy"])(output_node)

    project_name = '_'.join(['automodel', args.dataset, *[str(fe) for fe in args.feature_extrators]])

    clf = ak.AutoModel(
        project_name=project_name,
        inputs=input_node,
        outputs=output_node,
        objective='val_binary_accuracy',
        overwrite=True,
        max_trials=args.trials
    )
    clf.fit(
        x_tr, data['tr'][1],
        validation_data=(x_val, data['val'][1]),
        epochs=2500,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
        ],
        verbose=1
    )

    model = clf.export_model()
    model.summary()

    model_fpath = osp.join('model_autokeras', args.dataset, *[str(fe) for fe in args.feature_extrators])
    model.save(model_fpath)

    loaded_model = tf.keras.models.load_model(model_fpath, custom_objects=ak.CUSTOM_OBJECTS)

    h = clf.evaluate(x_inf, data['inf'][1])
    print(h)