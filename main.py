from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import matplotlib.pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-4-2-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print(nes_py)

done = True
for step in range(100000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()

env.close()