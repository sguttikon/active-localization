# active-localization

Rl Algorithms - [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

#### Steps:

- running localization agent
  1. run the gazebo simulation (with amcl, rviz, etc.) using command === ~$ roslaunch localization gazebo_localization.launch ===
  2. load the samele pretrained agent (ppo_turtlebot3_localize.zip) and run the localization code using command === ~$ python localization_rl_agent.py --file_path="../pretrained_agents/ppo_turtlebot3_localize" ===
  3. view the tensorboard logs === ~$ tensorboard --logdir ./ppo_localize_tensorboard/ ===
