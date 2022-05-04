import mat73, json, scipy.io, os
import os.path as osp
import pandas as pd
import numpy as np

def load_skab(fpath):
    df = pd.read_csv(fpath, delimiter=';')
    keys = [item for item in df.keys()]
    label_key = 'anomaly'
    other_keys = ['changepoint']
    assert label_key in keys
    feature_keys = [key for key in keys if key not in other_keys and key != label_key]
    X = df[feature_keys].values
    Y = df[label_key].values.reshape(-1, 1)
    return X, Y

def load_odds(fpath):
    try:
        mat = scipy.io.loadmat(fpath)
    except NotImplementedError:
        mat = mat73.loadmat(fpath)
    except:
        with open(fpath, 'r') as fp:
            lines = fp.readlines()
        lines = [line.strip() for line in lines]
        data = False
        categorical = []
        for line in lines:
            if not data:
                if "@attribute" in line:
                    attri = line.split()
                    name =
                    potential_ = attri[attri.index("@attribute") + 2]
                elif "@data" in line:
                    data = True
            else:
                newContent.append(line)

    X = mat['X']
    Y = mat['y']
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return X, Y

if __name__ == '__main__':

    data_dir = 'data'
    dataset_files = os.listdir(data_dir)

    with open('data/metainfo.json', 'r') as fp:
        metainfo = json.load(fp)

    for dataset in metainfo:
        print(dataset)
        fpath = osp.join(data_dir, dataset['file'])
        print(fpath)
        loader = locals()[f"load_{dataset['benchmark']}"]
        print(loader)
        X, Y = loader(fpath)
        print(fpath, X.shape, Y.shape)




