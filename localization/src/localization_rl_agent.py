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
import datetime

import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

def train_network(env, file_path: str, agent: str = 'PPO'):
    """
    Train the RL agent for localization task and store the policy/agent

    :params env: openai gym (TurtleBot3LocalizeEnv) instance
            str file_path: location to store the trained agent
            agent: stable_baselines3 agent to be used for training
    """
    dt_str = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')

    if agent == 'PPO':
        model = stable_baselines3.PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    else:
        return

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/checkpoints/", name_prefix=dt_str + 'rl_model')
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)

    # create the callback listeners list
    callback_list = CallbackList([eval_callback])

    model.learn(total_timesteps=30000, callback=callback_list, tb_log_name=dt_str + '_run')

    model.save(file_path)
    print('training finished')

def eval_network(env, file_path: str, agent: str = 'PPO'):
    """
    Evaluate the pretrained RL agent for localization task

    :params env: openai gym (TurtleBot3LocalizeEnv) instance
            str file_path: location to load the pretrained agent
            agent: stable_baselines3 agent to be used for evaluation
    """
    if agent == 'PPO':
        model = stable_baselines3.PPO.load(file_path)
    else:
        return

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
                    required=False, help='full path for location to store/load agent', \
                    default='./ppo_turtlebot3_localize')
    parser.add_argument('--train', dest='is_train', required=False, \
                    default=True, help='whether to train the agent')
    args = parser.parse_args()

    # create a new ros node
    rospy.init_node('turtlebot3_localization')

    # create a new gym turtlebot3 localization environment
    env = gym.make('TurtleBot3Localize-v0')

    # check out environment follows the gym interface
    #check_env(env)

    if args.is_train:
        train_network(env, args.file_path)
    eval_network(env, args.file_path)

    # prevent te code from exiting until an shutdown signal (ctrl+c) is received
    rospy.spin()
