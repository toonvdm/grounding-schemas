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

from typing import List
from jaxtyping import Array

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx

from pymdp import control
from pymdp.agent import Agent
from pymdp import inference, learning

import jax.nn as nn

from abcd.aif import free_energy_jax, generate_policies


class Buffer(eqx.Module):
    x: Array
    a: Array
    o: Array

    def __init__(self, buffer_size, obs_dim_size, state_dim_size, action_dim_size):
        self.x = jnp.zeros((buffer_size, state_dim_size))
        self.a = jnp.ones((buffer_size, action_dim_size)).astype(jnp.int32)
        self.o = jnp.zeros((buffer_size, obs_dim_size))

    def add(self, state, obs_idx, action):
        x = jnp.roll(self.x, -1, axis=0).at[-1].set(state)
        a = jnp.roll(self.a, -1).at[-1, action].set(1)
        o = jnp.roll(self.o, -1).at[-1, :].set(0).at[-1, obs_idx].set(1.0)
        return eqx.tree_at(lambda b: (b.x, b.a, b.o), self, (x, a, o))


def learn_b_smooth_fn(buffer, pymdp_agent):
    beliefs, actions = [buffer.x[:, None]], buffer.a[:-1]
    _, smooth_joint = jax.jit(inference.smoothing_ovf)(beliefs, pymdp_agent.B, actions)

    print(buffer.a[:-1].shape)
    qB, E_qB = learning.update_state_transition_dirichlet_f(
        pymdp_agent.pB[0][0],
        actions_f=buffer.a[:-1],
        joint_qs_f=smooth_joint[0][:, 0],
        lr=1.0,
    )
    pymdp_agent = eqx.tree_at(
        lambda x: (x.B, x.pB), pymdp_agent, ([E_qB[None]], [qB[None]])
    )
    return buffer, pymdp_agent


def learn_b_filter_fn(buffer, pymdp_agent):
    qs_prev, qs = buffer.x[-2], buffer.x[-1]
    obs = [buffer.o[-1].argmax()[None, None]]

    # When using filtering
    beliefs_B = jtu.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=1),
        [qs_prev[None, None]],
        [qs[None, None]],
    )
    pymdp_agent = pymdp_agent.infer_parameters(
        qs, obs, jnp.zeros((1, 1, 1)), beliefs_B=beliefs_B, lr_pB=1.0
    )

    return buffer, pymdp_agent


def add_death(mat, eps=1e-15):
    """
    Given an "illigal" non-stay action, there is eps chance to die.
    In the paper this is referred to as a dummy state.
    """
    for action in range(mat.shape[-1]):
        zer = np.where(mat[..., action].sum(axis=0) == 0)[0]
        mat[-1, zer, action] = eps
    # Stay dead once reached
    mat[-1, -1, :] = 1.0
    return mat


def add_stationary(mat, eps=1e-15):
    """
    Add an action to stand still
    """
    return np.concatenate(
        [mat, np.eye(mat.shape[0]).reshape(*mat[..., :1].shape)], axis=2
    )


def extract_AB(
    chmm,
    reduce=False,
    do_add_stationary=False,
    do_add_death=True,
    identity_prior=False,
    active_obs=None,
):
    death = int(do_add_death)
    n_obs = len(chmm.n_clones) + death

    T = chmm.T.transpose(2, 1, 0)

    unreduced_n_states = T.shape[0]
    if reduce:
        v = T.sum(axis=2).sum(axis=1).nonzero()[0]
        T = T[v, :][:, v]

    # Transition matrix
    B = np.zeros((T.shape[0] + death, T.shape[0] + death, T.shape[2]))
    B[: T.shape[0], : T.shape[1]] = T
    if do_add_death:
        B = add_death(B)
    else:
        B = B + 1e-12
    if do_add_stationary:
        B = add_stationary(B)

    if identity_prior:
        B += np.eye(B.shape[1])[..., None] * 0.1

    # Identity prior for unobserved states/unlearned state transitions
    B /= B.sum(axis=0, keepdims=True)

    # A matrix = likelihood matrix, unreduced matrix and uniform probabilities
    state_loc = np.hstack(
        (np.array([0], dtype=chmm.n_clones.dtype), chmm.n_clones)
    ).cumsum()

    A = np.zeros((n_obs, unreduced_n_states + death))
    for i in range(n_obs - int(death)):
        s, f = state_loc[i : i + 2]
        A[i, s:f] = 1.0

    # Direct mapping of state to a death observation
    if do_add_death:
        A[-1, -1] = 1.0

    if reduce and do_add_death:
        v = np.concatenate([v, np.array([-1])])

    # Only consider the reduced states
    if reduce:
        A = A[:, v]

    if active_obs is not None:
        A = A[active_obs]

    # Normalize over state: Sum_s P(o|s) = 1
    A += 1e-12
    A /= A.sum(axis=0, keepdims=True)

    return A, B, v


class CloneAgent(eqx.Module):
    pymdp_agent: eqx.Module
    empirical_prior: List[Array]

    buffer: Buffer

    use_smoothing: bool = eqx.field(static=True)

    def __init__(
        self,
        likelihood,
        transition,
        horizon_length=3,
        inductive=False,
        use_smoothing=False,
    ):
        if inductive:
            H = [jnp.zeros(likelihood[0].shape)]
            I = jax.vmap(partial(control.generate_I_matrix, depth=40))(
                [H[0][None]], [transition[None]], jnp.array([0.1])
            )
        else:
            H, I = None, None

        self.pymdp_agent = Agent(
            A=[likelihood],
            B=[transition],
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            H=H,
            I=I,
            learn_A=False,
            learn_B=False,
            action_selection="stochastic",
            gamma=1,
            policy_len=1,
            policies=generate_policies(horizon_length, transition.shape[-1], 1),
            onehot_obs=False,
            apply_batch=True,
            use_inductive=inductive,
            inductive_depth=40,
        )

        self.empirical_prior = self.pymdp_agent.D

        # Small state buffer, which defaults to just tracking the previous state for
        # parameter learning. Can be used for more timesteps when doing smoothing (10).
        self.buffer = (
            Buffer(
                2 if not use_smoothing else 10,
                state_dim_size=likelihood.shape[1],
                obs_dim_size=likelihood.shape[0],
                action_dim_size=transition.shape[-1],
            ).add(self.pymdp_agent.D[0].flatten(), 0, 0),
        )

        self.use_smoothing = use_smoothing

    @staticmethod
    def from_cscg(
        cscg,
        reduce=True,
        death_state=True,
        horizon_length=3,
        identity_prior=False,
        inductive=False,
        active_obs=None,
        sort_states=False,
        use_smoothing=False,
    ):
        if hasattr(cscg, "_inv_map"):
            # Basically, the inv map are the states of the lower level clone graph, that
            # were used. So for computational reasons, we only store that and expand
            # here
            mat_T = cscg.T.transpose(2, 1, 0)
            likelihood = np.zeros(
                (len(active_obs), cscg.n_clones[0] * len(cscg._inv_map))
            )

            nc = cscg.n_clones[0]
            for j, i in enumerate(cscg._inv_map):
                o = jnp.where(active_obs == (i))[0][0]
                likelihood[o, j * nc : (j + 1) * nc] = 1.0

            v = mat_T.sum(axis=2).sum(axis=1).nonzero()[0]
            likelihood = likelihood[:, v] + 1e-8
            likelihood /= likelihood.sum(axis=0, keepdims=True)

            _, transition, active_states_map = extract_AB(
                cscg,
                reduce,
                False,
                death_state,
                identity_prior=identity_prior,
                active_obs=None,
            )

            if sort_states:
                # Basically, we sort such that states with clones are at the end of the state
                # space. This essentially encodes the "start" of the schema as _not_ an
                # alternating state
                locs, states = jnp.where(likelihood > 1e-3)
                ulocs, counts = jnp.unique(locs, return_counts=True)
                full_counts = counts[jnp.searchsorted(ulocs, locs)]
                new_state_order = states[full_counts.argsort()].tolist()
                likelihood = likelihood[:, new_state_order]

                transition = transition[new_state_order, ...][:, new_state_order, :]

        else:
            likelihood, transition, active_states_map = extract_AB(
                cscg,
                reduce,
                False,
                death_state,
                identity_prior=identity_prior,
                active_obs=active_obs,
            )

        return CloneAgent(
            likelihood,
            transition,
            horizon_length,
            inductive=inductive,
            use_smoothing=use_smoothing,
        ), active_states_map

    @property
    def n_states(self):
        return self.pymdp_agent.B[0].shape[1]

    def _predict(self, qs, action):
        qs_last = jtu.tree_map(lambda x: x[:, -1], qs)
        propagate_beliefs = partial(
            control.compute_expected_state,
            B_dependencies=self.pymdp_agent.B_dependencies,
        )
        return jax.vmap(propagate_beliefs)(qs_last, self.pymdp_agent.B, action)

    @jax.jit
    def set_state_preference(self, target_state):
        new_C = jtu.tree_map(lambda x: x * 0, self.pymdp_agent.C)
        new_H = jtu.tree_map(lambda x: x * 0, self.pymdp_agent.H)

        new_H[0] = new_H[0].at[:, target_state].set(5.0)

        pa = self.pymdp_agent
        new_I = jax.vmap(partial(control.generate_I_matrix, depth=pa.inductive_depth))(
            new_H, pa.B, pa.inductive_threshold
        )

        pa = eqx.tree_at(lambda x: (x.C, x.H, x.I), pa, (new_C, new_H, new_I))
        new_agent = eqx.tree_at(lambda x: x.pymdp_agent, self, pa)
        return new_agent

    @partial(jax.jit, static_argnames=["inhibit"])
    def set_preference(self, target_observation, inhibit=None):
        new_C = jtu.tree_map(lambda x: x * 0, self.pymdp_agent.C)
        new_C[0] = new_C[0].at[..., target_observation].set(100.0)
        if inhibit is not None:
            new_C[0] = new_C[0].at[..., inhibit].set(-10.0)

        pymdp_agent = eqx.tree_at(lambda x: x.C, self.pymdp_agent, new_C)

        new_agent = eqx.tree_at(lambda x: x.pymdp_agent, self, pymdp_agent)
        return new_agent

    def infer_state(self, obs):
        pymdp_agent = jtu.tree_map(lambda x: x, self.pymdp_agent)
        qs = pymdp_agent.infer_states(
            observations=obs, empirical_prior=self.empirical_prior
        )

        buffer = jtu.tree_map(lambda x: x, self.buffer)
        if pymdp_agent.learn_A or pymdp_agent.learn_B:
            # TODO: properly track actions.
            buffer = buffer.add(qs[0][0, 0], obs_idx=obs[0][0, 0], action=0)
            if self.use_smoothing:
                # skip until the buffer is filled
                buffer, pymdp_agent = jax.lax.cond(
                    buffer.x[0].sum() > 0,
                    learn_b_smooth_fn,
                    lambda *args: args,
                    *(buffer, pymdp_agent),
                )
            else:
                buffer, pymdp_agent = learn_b_filter_fn(buffer, pymdp_agent)

        new_agent = eqx.tree_at(lambda x: x.pymdp_agent, self, pymdp_agent)
        new_agent = eqx.tree_at(lambda x: x.buffer, new_agent, buffer)
        return new_agent, qs

    @jax.jit
    def infer_action(self, qs, key):
        key, subkey = jr.split(key)
        qpi, G = self.pymdp_agent.infer_policies(qs)
        action = self.pymdp_agent.sample_action(qpi, rng_key=subkey[None])
        agent = eqx.tree_at(
            lambda x: x.empirical_prior, self, self._predict(qs, action)
        )
        return agent, action, {"qpi": qpi, "G": G}

    @jax.jit
    def free_energy_state(self, qs):
        return free_energy_jax(qs[0][0][0], self.pymdp_agent.H[0][0])[0]

    @partial(jax.jit, static_argnames=["prior"])
    def free_energy(self, qs, prior=None):
        if prior is None:
            prior = self.pymdp_agent.C[0][0]
            prior = prior @ self.pymdp_agent.A[0][0]

        return free_energy_jax(qs[0][0][0], prior)[0]

    @partial(jax.jit, static_argnames=["sample"])
    def empirical_prior_to_obs(self, key, sample=False):
        if sample:
            key, subkey = jr.split(key)
            p = self.empirical_prior[0][0]
            p = p * (p > 1e-3)  # Remove tiny contributions
            p = p / p.sum()
            ep_idx = jr.choice(subkey, np.arange(p.shape[0]), p=p)
        else:
            ep_idx = self.empirical_prior[0][0].argmax()

        return self.pymdp_agent.A[0][0, :, ep_idx]
