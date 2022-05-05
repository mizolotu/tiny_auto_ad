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
        label_key = 'class'
        mat = {'X': [], 'y': []}
        for line in lines:
            if not data:
                if "@attribute" in line:
                    attri = line.split()
                    attr_name = attri[attri.index("@attribute") + 1]
                    value_set = ''.join(attri[attri.index("@attribute") + 2:])
                    if value_set.startswith('{') and value_set.endswith('}'):
                        value_set = sorted(value_set.split('{')[1].split('}')[0].split(','))
                        if attr_name == label_key:
                            assert len(value_set) == 2
                            label_position = len(categorical)
                        categorical.append(value_set)
                    else:
                        categorical.append(None)
                elif "@data" in line:
                    data = True
            else:
                x = []
                for i, (item, c) in enumerate(zip(line.split(','), categorical)):
                    if i == label_position:
                        y = c.index(item)
                    elif c is not None:
                        x_ = np.zeros(len(c))
                        x_[c.index(item)] = 1
                        x.append(x_)
                    else:
                        x.append(float(item))
                mat['X'].append(np.hstack(x))
                mat['y'].append(y)
        mat['X'] = np.vstack(mat['X'])
        mat['y'] = np.hstack(mat['y']).reshape(-1, 1)

    X = mat['X']
    Y = mat['y']
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return X, Y

if __name__ == '__main__':

    data_dir = 'data'
    dataset_files = os.listdir(data_dir)

    with open(f'{data_dir}/metainfo.json', 'r') as fp:
        metainfo = json.load(fp)

    for dataset in metainfo:
        fpath = osp.join(data_dir, dataset['file'])
        loader = globals()[f"load_{dataset['benchmark']}"]
        X, Y = loader(fpath)
        print(fpath, X.shape, Y.shape)




