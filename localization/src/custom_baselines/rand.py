#!/usr/bin/env python3

from typing import Union, Optional, Tuple, Iterable
import time
import io
import pathlib
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

class RAND(BaseAlgorithm):
    """
    Random behaviour
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        n_steps: int = 2048,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0
    ):
        """

        :param env: The environment to learn from (if registered in Gym, can be str)
        :param n_steps: The number of steps to run for each environment per update
               (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
        :param verbose: the verbosity level: 0 not output, 1 info, 2 debug
        """
        super(RAND, self).__init__(
            policy=BasePolicy,
            env=env,
            policy_base=BasePolicy,
            learning_rate=0.0,
            tensorboard_log=tensorboard_log,
            verbose=verbose
        )
        self.n_steps = n_steps

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_rollout_steps: int
    ) -> bool:
        """
        Collect rollout using the current policy

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_steps: Number of experiences to collect per environment

        :return bool: True
        """
        n_steps = 0

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            actions = [env.action_space.sample()]

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            callback.on_step()

            self._update_info_buffer(infos)
            n_steps += 1

            self._last_obs = new_obs
            self._last_dones = dones

        callback.on_rollout_end()

        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = 'RAND',
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True
    ) -> 'RAND':

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(self.env, callback, self.n_steps)

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

        callback.on_training_end()

        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        actions = [self.env.action_space.sample()]
        return actions, state

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None
    ) -> None:
        pass    # do nothing

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = 'auto',
        **kwargs
    ) -> 'BaseAlgorithm':
        model = cls(
            env=env
        )
        model._setup_model()

        return model
