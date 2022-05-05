import gym

from stable_baselines3 import PPO

if __name__ == '__main__':

    env = gym.make("CartPole-v1")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()