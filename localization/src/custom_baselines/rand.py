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
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import safe_mean

class RAND(BaseAlgorithm):
    """
    Random behaviour
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        tensorboard_log: Optional[str] = None,
        verbose: int = 0
    ):
        """

        :param env: The environment to learn from (if registered in Gym, can be str)
               tensorboard_log: the log location for tensorboard (if None, no logging)
               verbose: the verbosity level: 0 not output, 1 info, 2 debug
        """
        super(RAND, self).__init__(
            policy=BasePolicy,
            env=env,
            policy_base=BasePolicy,
            learning_rate=0.0,
            tensorboard_log=tensorboard_log,
            verbose=verbose
        )

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

        while self.num_timesteps < total_timesteps:
            # collect rollouts
            actions = [self.env.action_space.sample()]
            new_obs, rewards, dones, infos = self.env.step(actions)

            self.num_timesteps += self.env.num_envs
            self._update_info_buffer(infos)
            self._last_obs = new_obs
            self._last_dones = dones

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
