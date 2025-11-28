# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/toonvdm/grounding-schemas/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import jax.nn as nn

from jaxtyping import Array

import equinox as eqx

from pymdp.maths import log_stable

from abcd.utils import lr2obs, obs2lr
from abcd.agent.clone_agent import CloneAgent
from abcd.agent.grounding import GroundingLikelihood
from abcd.agent import grounding as grounding_tools
from abcd.aif import compute_param_infogain


def modify_reward_likelihood(agent):
    likelihood = agent.pymdp_agent.A[0][0]
    rew_obs, states = jnp.where(likelihood > 0.5)
    for s, r in zip(states, rew_obs):
        clone_states = jnp.where(likelihood[r] > 0.25)[0].tolist()
        other_states = jnp.array([i for i in states.tolist() if i not in clone_states])
        nro = lr2obs(obs2lr(r)[0], 0)
        likelihood = likelihood.at[nro, other_states].set(1).at[nro, s].set(0)

    likelihood = likelihood / likelihood.sum(axis=1, keepdims=True)

    pymdp_agent = eqx.tree_at(lambda x: x.A, agent.pymdp_agent, [likelihood[None]])
    agent = eqx.tree_at(lambda x: x.pymdp_agent, agent, pymdp_agent)
    return agent


def hierarchical_comm(agent, high, low, lo_qs, qz, reward, key):
    """
    When there is hierarchical communications, we infer the high level state through
    bottom-up inference, and set a top-down preference
    """
    key, *subkeys = jr.split(key, 3)
    # If the free energy is low, the agent has reached its internal goal
    if agent._remap:

        def true_fn(high):
            lo_state = lo_qs[0][0, 0].argmax()
            remapped_loc = grounding_tools.remap(agent.grounding, qz, lo_state)
            hi_obs = [lr2obs(remapped_loc, 1)[None, None]]
            high, hi_qs = high.infer_state(hi_obs)
            high, _, _ = high.infer_action(hi_qs, subkeys[0])
            return high, hi_qs

        def false_fn(high):
            hi_qs = [agent.high_qs[0][None]]
            return high, hi_qs

        high, hi_qs = jax.lax.cond(reward, true_fn, false_fn, high)
    else:
        hi_obs = agent.state_to_observation(lo_qs, reward)
        high, hi_qs = high.infer_state(hi_obs)
        high, _, _ = high.infer_action(hi_qs, subkeys[0])

    agent = eqx.tree_at(lambda x: x.high, agent, high)

    # This maps the empirical prior to an action
    hi_act = agent.get_high_action(
        qz, subkeys[1], random=agent._random_high, remap=agent._remap
    )
    low = low.set_state_preference(hi_act)

    return (agent, high, low, hi_qs)


def no_hierarchical_comm(agent, high, low, lo_qs, qz, reward, key):
    """
    When there is no hierarchical communication, we just pass through the previous
    preference/models
    """
    return agent, high, low, [agent.high_qs[0][None]]


class HierarchicalAgent(eqx.Module):
    low: CloneAgent
    high: CloneAgent
    grounding: GroundingLikelihood
    high_qs: Array

    # Static variables that just do a configuration
    _sample_preference: bool = eqx.field(static=True)
    _ell_threshold: float = eqx.field(static=True)
    _free_energy_threshold: float = eqx.field(static=True)
    _num_locations: int = eqx.field(static=True)
    _random_high: bool = eqx.field(static=True)
    _learn: bool = eqx.field(static=True)
    _remap: bool = eqx.field(static=True)

    def __init__(
        self,
        low: CloneAgent,
        high: CloneAgent,
        remap=False,
        fe_treshold=-0.05,
        random_high=False,
        use_mixture=False,
        seed=0,
        learn=True,
        ell_treshold=-2.5,
        sample_preference=False,
        gl_gamma=1.0,
    ):
        low = low.set_state_preference(0)
        self.low = low

        self._ell_threshold = ell_treshold
        self._sample_preference = sample_preference

        num_locations = low.pymdp_agent.A[0][0].shape[1]

        # set the top-down preferences to get reward
        high = high.set_preference(self.get_rewarding_states(num_locations))
        if not remap and not high.pymdp_agent.learn_B:
            high = modify_reward_likelihood(high)

        self.high = high
        self.high_qs = high.pymdp_agent.D

        self._free_energy_threshold = fe_treshold
        self._remap = remap
        self._num_locations = num_locations

        num_clusters = 1 if not use_mixture else 75
        self.grounding = GroundingLikelihood(
            num_locations, num_clusters, seed=seed, gamma=gl_gamma
        )

        self._random_high = random_high
        self._learn = learn

    def get_rewarding_states(self, num_locs=9):
        rewarding_states = []
        for i in range(num_locs):
            rewarding_states.append(lr2obs(i, 1))
        rewarding_states = np.asarray(rewarding_states).astype(np.int32)
        rewarding_states = rewarding_states[rewarding_states >= 0]
        return rewarding_states

    def state_to_observation(self, qs_low, reward):
        return jtu.tree_map(
            lambda x: lr2obs(x[0, 0].argmax(), reward)[None, None], qs_low
        )

    @partial(jax.jit, static_argnames=["random", "remap"])
    def get_high_action(self, qz, key, random=False, remap=False):
        key, *subkeys = jr.split(key, 3)
        if random:
            action = jr.randint(
                subkeys[0], minval=0, maxval=self.grounding.prior.shape[1], shape=(1,)
            )
        else:
            o = self.high.empirical_prior_to_obs(
                key=subkeys[0], sample=self._sample_preference
            ).argmax()
            action = obs2lr(o)[0]

            if remap:
                action = grounding_tools.inv_remap(
                    self.grounding, qz, action, key=subkeys[1]
                )

        return action

    @jax.jit
    def get_low_action(self, qs, key):
        key, *subkeys = jr.split(key, 3)

        qpi, G = self.low.pymdp_agent.infer_policies(qs)

        action = self.low.pymdp_agent.sample_action(qpi, rng_key=subkeys[0][None])

        new_low_agent = eqx.tree_at(
            lambda x: x.empirical_prior, self.low, self.low._predict(qs, action)
        )
        new_agent = eqx.tree_at(lambda x: x.low, self, new_low_agent)

        return new_agent, action, {"qpi": qpi, "G": G}

    @jax.jit
    def remap_fn(self, lo_qs, reward, key):
        grounding = jtu.tree_map(lambda x: x, self.grounding)
        new_agent = jtu.tree_map(lambda x: x, self)

        ep_obs = self.high.empirical_prior_to_obs(key, sample=False).argmax()

        # Map it to a one-hot top-down observation on location
        td_obs = nn.one_hot(obs2lr(ep_obs)[0], self._num_locations)

        # Bottom-up message: where did we find reward
        bu_obs = lo_qs[0][0, 0]
        grounding, max_ell, qz = grounding_tools.infer_qz(
            grounding, td_obs, bu_obs, reward
        )
        new_agent = eqx.tree_at(lambda x: x.grounding, new_agent, grounding)

        if self._learn:
            # Only allow growth once it accumulated some counts
            should_grow = grounding.prior[qz.argmax()].max() > 5.0

            grounding, qz = jax.lax.cond(
                (max_ell < self._ell_threshold) * should_grow,
                grounding_tools.add_cluster,
                lambda g, q: (g, q),
                *(grounding, qz),
            )
            lr, bu_obs = jax.lax.cond(
                reward,
                lambda: (1.0, bu_obs),
                lambda: (0.05, (1 - bu_obs) / (1 - bu_obs).sum()),
            )
            # jax.debug.print("{r} {lr}", r=reward, lr=lr)
            grounding = grounding_tools.update(
                grounding, td_obs, bu_obs, lr=lr, fr=0.00, qz=qz
            )

        new_agent = eqx.tree_at(lambda x: x.grounding, new_agent, grounding)
        return new_agent, qz, max_ell

    @jax.jit
    def act(self, obs, reward, key):
        key, *subkeys = jr.split(key, 4)

        # Variables that have to be updated at the end
        high = jtu.tree_map(lambda x: x, self.high)
        low = jtu.tree_map(lambda x: x, self.low)
        new_agent = jtu.tree_map(lambda x: x, self)

        # Bottom-up inference
        low, lo_qs = low.infer_state(jnp.expand_dims(jnp.asarray([obs]), [0, 1]))

        # Learn at every step
        qz, max_ell = jnp.array([1]), jnp.array([1])
        if self._remap:
            new_agent, qz, max_ell = new_agent.remap_fn(lo_qs, reward, key=subkeys[1])

        fe = low.free_energy_state(lo_qs)

        new_agent, high, low, hi_qs = jax.lax.cond(
            reward | (fe <= self._free_energy_threshold)[0],
            hierarchical_comm,
            no_hierarchical_comm,
            *(new_agent, high, low, lo_qs, qz, reward, subkeys[0]),
        )

        # Update our agent with the new values
        new_agent = eqx.tree_at(lambda x: x.high_qs, new_agent, [hi_qs[0][0]])
        new_agent = eqx.tree_at(lambda x: x.low, new_agent, low)
        new_agent = eqx.tree_at(lambda x: x.high, new_agent, high)

        # Get the low level action
        new_agent, lo_act, low_info = new_agent.get_low_action(lo_qs, subkeys[1])

        # Some variables we want to track
        info = {
            "max_ell": max_ell,
            "qz": qz,
            "fe": fe,
            "low_state": lo_qs,
            "high_state": hi_qs,
            "high_empirical_prior": high.empirical_prior[0],
            "qpi_low": low_info["qpi"],
            "G_low": low_info["G"],
        }
        return new_agent, lo_act[0][0].astype(jnp.int32), info
