#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_reward(
    time_steps: np.ndarray,
    avg_rewards: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot avg_reward of #episodes per evaluation step against time steps
    """

    fig, ax = plt.subplots()
    ax.plot(time_steps, avg_rewards, marker='o')
    ax.plot(time_steps, np.full(avg_rewards.shape, 25),'g--')

    ax.set(xlabel='timesteps', ylabel='avg reward of #episodes', title='Average Reward')
    ax.grid()

    if save_path is not None:
        fig.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--log_dir', dest='log_dir', \
                        required=False, help='full path for location to .npz file', \
                        default='./logs/RAND/results/evaluations.npz')
    args = parser.parse_args()
    npz_data = np.load(args.log_dir)
    time_steps = npz_data['timesteps']
    avg_rewards = np.mean(npz_data['results'], axis=1)
    avg_ep_lengths = np.mean(npz_data['ep_lengths'], axis=1)

    plot_reward(time_steps, avg_rewards)
