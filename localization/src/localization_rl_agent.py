#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'openai_ros' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/openai-rosbot-env/openai_ros/src')
from openai_ros.task_envs.turtlebot3 import turtlebot3_localize
import gym
import rospy

from stable_baselines3 import PPO

if __name__ == '__main__':

    # create a new ros node
    rospy.init_node('turtlebot3_localization')

    # create a new gym turtlebot3 localization environment
    env = gym.make('TurtleBot3Localize-v0')

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()

    # prevent te code from exiting until an shutdown signal (ctrl+c) is received
    rospy.spin()
