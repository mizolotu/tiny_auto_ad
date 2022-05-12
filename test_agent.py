import numpy as np
import argparse as arp

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from auto_ad_env import AutoAdEnv
from preprocess_data import load_dataset
from matplotlib import pyplot as pp

from train_agent import make_env, TE_DATA_DIR, SEED

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test agent.')
    parser.add_argument('-i', '--index', help='Model index', type=int, default=100*4*10)
    args = parser.parse_args()

    target_dataset = load_dataset(TE_DATA_DIR, series_len=32, labels={0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']})
    n_features = target_dataset['X'].shape[1]

    env_class = AutoAdEnv
    n_steps = 100
    model_idx = args.index

    te_env = SubprocVecEnv([make_env(env_class, SEED, 0, [target_dataset], n_steps, n_features)])

    print('Random policy:')

    obs = te_env.reset()
    rewards = []
    for i in range(n_steps):
        action = [te_env.action_space.sample()]
        obs, reward, dones, info = te_env.step(action)
        rewards.extend(reward)
    print(f'Mean reward: {np.mean(rewards)}, max reward: {np.max(rewards)}')
    pp.plot(np.array(rewards), label='Random policy')

    model = PPO.load(f'logs/rl_model_{model_idx}_steps.zip', env=te_env)

    print('Learned policy:')

    obs = te_env.reset()
    rewards = []
    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = te_env.step(action)
        rewards.extend(reward)
    print(f'Mean reward: {np.mean(rewards)}, max reward: {np.max(rewards)}')
    pp.plot(np.array(rewards), label='Learned policy')

    pp.savefig('random_vs_learned.png')




