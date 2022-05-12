import json, gym
import sys
import numpy as np
import os.path as osp
import argparse as arp

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from auto_ad_env import AutoAdEnv
from typing import Callable
from preprocess_data import load_skab, load_odds, load_dataset
from stable_baselines3.common.callbacks import CheckpointCallback

TR_DATA_DIR = 'data/benchmarks'
TE_DATA_DIR = 'data/adxl_fan'
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

    parser = arp.ArgumentParser(description='Train agent.')
    parser.add_argument('-i', '--iterations', help='Number of training iterations', type=int, default=1000)
    parser.add_argument('-e', '--environments', help='Number of training environments', type=int, default=4)
    args = parser.parse_args()

    target_dataset = load_dataset(TE_DATA_DIR, series_len=32, labels={0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']})
    n_features = target_dataset['X'].shape[1]

    with open(f'{TR_DATA_DIR}/metainfo.json', 'r') as fp:
        datasets = json.load(fp)

    datasets_selected = []
    for dataset in datasets:
        fpath = osp.join(TR_DATA_DIR, dataset['file'])
        X, Y = globals()[f"load_{dataset['benchmark']}"](fpath)
        if X.shape[1] >= n_features:
            datasets_selected.append({'X': X, 'Y': Y})

    if len(datasets_selected) == 0:
        print('No training datasets with the number of features specified have been found!')
        sys.exit(1)

    idx = np.arange(len(datasets_selected))
    np.random.shuffle(idx)
    val, tr = np.split(idx, [int(VAL_SPLIT * len(idx))])
    tr_datasets = [datasets_selected[i] for i in tr] if len(tr) > 0 else datasets_selected
    val_datasets = [datasets_selected[i] for i in val] if len(val) > 0 else datasets_selected

    env_class = AutoAdEnv
    n_envs = args.environments
    n_steps = 100
    n_iterations = args.iterations
    eval_freq = 10

    train_env = SubprocVecEnv([make_env(env_class, SEED, i, tr_datasets, n_steps, n_features) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(env_class, SEED, n_envs, val_datasets, n_steps, n_features)])

    model = PPO("MlpPolicy", env=train_env, n_steps=n_steps, batch_size=n_steps, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq * n_steps, save_path='./logs/', name_prefix='rl_model')
    model.learn(
        total_timesteps=n_steps * n_envs * n_iterations,
        eval_env=eval_env,
        eval_freq=eval_freq * n_steps,
        callback=checkpoint_callback,
    )

    te_env = SubprocVecEnv([make_env(env_class, SEED, n_envs + 1, [target_dataset], n_steps, n_features)])

    print('Random policy:')
    obs = te_env.reset()
    rewards = []
    for i in range(n_steps):
        action = [te_env.action_space.sample()]
        obs, reward, dones, info = te_env.step(action)
        rewards.extend(reward)
    print(f'Mean reward: {np.mean(rewards)}, max reward: {np.max(rewards)}')

    PPO.load(f'logs/rl_model_{n_iterations * n_envs * n_steps}_steps.zip', env=te_env)

    print('Learned policy:')
    obs = te_env.reset()
    rewards = []
    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = te_env.step(action)
        rewards.extend(reward)
    print(f'Mean reward: {np.mean(rewards)}, max reward: {np.max(rewards)}')





