# -*- coding: utf-8 -*-
#
# Copyright 2020 Roberto Cipollone, Marco Favorito
#
# ------------------------------
#
# This file is part of multinav.
#
# multinav is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multinav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multinav.  If not, see <https://www.gnu.org/licenses/>.
#
"""Q-Learning implementation."""
import logging
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional

import gym
import numpy as np

from multinav.wrappers.utils import MyStatsRecorder

logger = logging.getLogger(__name__)


def _random_action(nb_actions):
    """Random actions."""
    return (
        np.random.randn(
            nb_actions,
        )
        * sys.float_info.epsilon
    )


def q_learning(
    env: MyStatsRecorder,
    total_timesteps: int = 2000000,
    alpha: float = 0.1,
    eps: float = 0.1,
    gamma: float = 0.9,
    learning_rate_decay: bool = False,
    epsilon_decay: bool = False,
    epsilon_end: float = 0.0,
    learning_rate_end: float = 0.0,
    stats_wrapper: Optional[MyStatsRecorder] = None,
    env_unshaped: Optional[gym.Env] = None,
) -> Dict[Any, np.ndarray]:
    """
    Learn a Q-function from a Gym env using vanilla Q-Learning.

    :param env: a gym environment. For convenience, we request to wrap it
        into a MyStatsRecorder so that additional metrics can be logged.
    :param env_unshaped: when using reward shaping, this must be the same as
        env, without reward shaping applied.
    :return the Q function: a dictionary from states to array of Q values for every action.
    """
    # Vars
    alpha0 = alpha
    eps0 = eps

    # Init
    nb_actions = env.action_space.n
    Q_act: Dict[Any, np.ndarray] = defaultdict(partial(_random_action, nb_actions))
    Q_unshaped: Dict[Any, np.ndarray] = defaultdict(partial(_random_action, nb_actions))

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q_act[state])

    done = True
    for step in range(total_timesteps):

        if done:
            state = env.reset()
            state_unsh = env_unshaped.reset()
            assert (state == state_unsh).all()  # Assumign numpy observations
            done = False

        # Step
        action = choose_action(state)
        state2, reward, done, _ = env.step(action)

        # Step (without reward shaping)
        if env_unshaped is not None:
            state2_unsh, reward_unsh, done_unsh, _ = env_unshaped.step(action)
            assert (state2 == state2_unsh).all()  # Assumign numpy observations
            assert done == done_unsh

        # Apply
        _q_learning_step(env, Q_act, state, state2, action, reward, gamma, alpha)
        if env_unshaped is not None:
            _q_learning_step(env_unshaped, Q_unshaped, state, state2, action, reward_unsh, gamma, alpha)

        state = state2

        # Decays
        if step % 10 == 0:
            frac = step / total_timesteps
            if learning_rate_decay:
                alpha = alpha0 * (1 - frac) + learning_rate_end * frac
            if epsilon_decay:
                eps = eps0 * (1 - frac) + epsilon_end * frac

            # Log
            print(" Eps:", round(eps, 3), end="\r")

    print()
    return (Q_act, Q_unshaped)


def _q_learning_step(
    log_env: MyStatsRecorder,
    Q: Dict[Any, np.ndarray],
    state: Any,
    state2: Any,
    action: int,
    reward: float,
    gamma: float,
    alpha: float,
):
    """Perform a single Q learning step."""
    # Compute update
    td_update = reward + gamma * np.max(Q[state2]) - Q[state][action]

    # Log
    logger.debug(
        f"Q[{state}][{action}] = {Q[state][action]} "
        f"-> {reward + gamma * np.max(Q[state2])}"
    )
    log_env.update_extras(tf=abs(td_update))

    # Apply
    Q[state][action] += alpha * td_update
