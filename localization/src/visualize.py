#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_reward(
    time_steps: np.ndarray,
    avg_rewards: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot avg_reward of #episodes per evaluation step against time steps
    """

    fig, ax = plt.subplots()
    ax.plot(time_steps, avg_rewards)

    ax.set(xlabel='timesteps', ylabel='avg reward of #episodes', title='Average Reward')
    ax.grid()

    if save_path is not None:
        fig.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    log_dir = './logs/evaluations.npz'
    npz_data = np.load(log_dir)
    time_steps = npz_data['timesteps']
    avg_rewards = np.mean(npz_data['results'], axis=1)
    avg_ep_lengths = np.mean(npz_data['ep_lengths'], axis=1)

    plot_reward(time_steps, avg_rewards)
