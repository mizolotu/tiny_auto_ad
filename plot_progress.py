import numpy as np
import pandas as pd

from matplotlib import pyplot as pp

if __name__ == '__main__':

    fpath = 'log.txt'
    with open(fpath, 'r') as f:
        lines = f.readlines()

    rewards = []
    timesteps = []
    found = False
    for line in lines:
        if found and 'mean_reward' in line:
        #if found and 'ep_rew_mean' in line:
            spl = line.split('|')
            rewards.append(float(spl[2]))
        if found and 'total_timesteps' in line:
            spl = line.split('|')
            timesteps.append(float(spl[2]))
            found = False
        if 'eval/' in line:
        #if 'rollout/' in line:
            found = True

    rewards = np.array(rewards)
    timesteps = np.array(timesteps)

    window_size = 10
    numbers_series = pd.Series(rewards)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = np.array(moving_averages_list[window_size - 1:])

    pp.plot(timesteps[window_size - 1:], final_list)
    pp.savefig('reward.png')
