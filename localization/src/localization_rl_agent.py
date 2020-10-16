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
import argparse

from stable_baselines3 import DQN
from stable_baselines3 import PPO

def train_network(file_path: str):
    """
    Train the RL agent for localization task and store the policy/agent

    :params str file_path: location to store the trained agent
    """
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25000)

    model.save(file_path)
    print('training finished')

def eval_network(file_path: str):
    """
    Evaluate the pretrained RL agent for localization task

    :params str file_path: location to load the pretrained agent
    """
    model = PPO.load(file_path)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
          break

    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train/Evaluate localization RL agent')
    parser.add_argument('--file_path', dest='file_path', \
                    required=True, help='full path for location to store/load agent')
    args = parser.parse_args()

    # create a new ros node
    rospy.init_node('turtlebot3_localization')

    # create a new gym turtlebot3 localization environment
    env = gym.make('TurtleBot3Localize-v0')

    #train_network(args.file_path)
    eval_network(args.file_path)

    # prevent te code from exiting until an shutdown signal (ctrl+c) is received
    rospy.spin()
