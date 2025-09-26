import gym

from stable_baselines3.common.env_util import make_vec_env

env = gym.make("CartPole-v1")

done = True
for step in range(100000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()

env.close()