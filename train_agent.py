import json, gym
import sys

import numpy as np
import os.path as osp

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from auto_ad_env import AutoAdEnv
from typing import Callable
from preprocess_data import load_skab, load_odds

DATA_DIR = 'data'
VAL_SPLIT = 0.3
SEED = 0

def make_env(env_class: Callable, seed: int, rank: int, *args) -> Callable:
    def _init() -> gym.Env:
        env = env_class(*args)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    with open(f'{DATA_DIR}/metainfo.json', 'r') as fp:
        datasets = json.load(fp)

    n_features = 12
    datasets_selected = []
    for dataset in datasets:
        fpath = osp.join(DATA_DIR, dataset['file'])
        X, Y = globals()[f"load_{dataset['benchmark']}"](fpath)
        if X.shape[1] >= n_features:
            datasets_selected.append({'X': X, 'Y': Y})

    if len(datasets_selected) == 0:
        print('No datasets found!')
        sys.exit(1)

    idx = np.arange(len(datasets_selected))
    np.random.shuffle(idx)
    val, tr = np.split(idx, [int(VAL_SPLIT * len(idx))])
    tr_datasets = [datasets_selected[i] for i in tr] if len(tr) > 0 else datasets_selected
    val_datasets = [datasets_selected[i] for i in val] if len(val) > 0 else datasets_selected

    env_class = AutoAdEnv
    n_tr = 1
    n_steps = 32
    n_features = 12

    train_env = SubprocVecEnv([make_env(env_class, SEED, i, tr_datasets, n_steps, n_features) for i in range(n_tr)])
    eval_env = SubprocVecEnv([make_env(env_class, SEED, n_tr, val_datasets, n_steps, n_features)])

    model = PPO("MlpPolicy", env=train_env, n_steps=32, verbose=1)
    model.learn(total_timesteps=32*4*1000, eval_env=eval_env)