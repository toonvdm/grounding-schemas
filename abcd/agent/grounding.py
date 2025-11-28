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

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import jax.nn as nn
from jaxtyping import Array
import equinox as eqx
from tensorflow_probability.substrates.jax.distributions import DirichletMultinomial

from pymdp.maths import multidimensional_outer


class GroundingLikelihood(eqx.Module):
    prior: Array
    B: Array
    used_mask: Array
    qz_prev: Array

    gamma: float = eqx.field(static=True)

    def __init__(self, n_obs, n_clusters=1, seed=0, gamma=1):
        self.prior = (
            jr.uniform(jr.PRNGKey(seed), (n_clusters, n_obs, n_obs)) * 0.001 + 0.001
        )
        # Identity transition model on the likelihood state
        mat_B = jnp.eye(n_clusters) * 0.999 + 0.001 / (n_clusters - 1 + 1e-8)
        self.B = mat_B / mat_B.sum(axis=-1, keepdims=True)
        # Initialise with a single cluster and belief over this clsuter
        self.used_mask = jnp.zeros(n_clusters).at[0].set(1.0)
        self.qz_prev = jtu.tree_map(lambda x: x, self.used_mask)
        self.gamma = gamma

    @property
    def expected_likelihood(self):
        return self.prior / self.prior.sum(axis=-1, keepdims=True)


def remap(glh, qz, bu_obs):
    return glh.expected_likelihood[qz.argmax(), :, bu_obs.astype(jnp.int32)].argmax()


def inv_remap(glh, qz, td_obs, key):
    # return glh.expected_likelihood[qz.argmax(), td_obs.astype(jnp.int32), :].argmax()
    p = nn.softmax(glh.gamma * glh.prior[qz.argmax(), td_obs.astype(jnp.int32), :])
    return jr.choice(key, jnp.arange(p.shape[0]), p=p)


@partial(jax.jit, static_argnames=["soft_update"])
def update(glh, td_msg, bu_msg, lr, fr, qz, soft_update=True):
    prior = glh.prior
    if soft_update:
        counts = multidimensional_outer([td_msg, bu_msg])
        prior = prior.at[qz.argmax()].set((1 - fr) * prior[qz.argmax()] + lr * counts)
    else:
        td_msg, bu_msg = td_msg.argmax(), bu_msg.argmax()
        idx = qz.argmax()
        prior = (1 - fr) * prior
        prior = prior.at[idx, td_msg, bu_msg].set(prior[idx, td_msg, bu_msg] + lr)

    return eqx.tree_at(lambda x: x.prior, glh, prior.clip(min=1e-8))


@jax.jit
def add_cluster(grounding, qz):
    grow = grounding.used_mask.sum() < grounding.used_mask.shape[0]

    new_cluster = grounding.used_mask.sum().astype(jnp.int32)

    used_mask = jax.lax.cond(
        grow,
        lambda: grounding.used_mask.at[new_cluster].set(1.0),
        lambda: grounding.used_mask,
    )
    qz = jax.lax.cond(
        grow, lambda: (used_mask * 0).at[new_cluster].set(1.0), lambda: qz
    )

    new_likelihood = eqx.tree_at(lambda x: x.used_mask, grounding, used_mask)
    new_likelihood = eqx.tree_at(lambda x: x.qz_prev, new_likelihood, qz)
    return new_likelihood, qz


@jax.jit
def infer_qz(glh, td_obs, bu_obs, reward):
    def step_fn(concentration):
        inv_prior = 1 - concentration / concentration.sum()
        inv_prior = inv_prior / inv_prior.sum()
        elogp = jax.lax.cond(
            reward,
            DirichletMultinomial(1.0, concentration).log_prob,
            DirichletMultinomial(1.0, (inv_prior * concentration.sum())).log_prob,
            bu_obs,
        )
        return elogp, concentration.max()

    # vmap over likelihoods to measure log likelihoods
    ells, concs = jax.vmap(step_fn)(glh.prior[:, td_obs.argmax(), :])

    # Integrate likelihood with empirical prior
    qz = nn.softmax(ells * glh.used_mask + (1 - glh.used_mask) * (-1e8))
    prior = glh.qz_prev @ glh.B
    qz = qz * prior
    qz = qz / (qz.sum() + 1e-8)

    # jax.debug.print("{qz}", qz=qz)

    c_i = concs[glh.qz_prev.argmax()]
    qz = jax.lax.cond(c_i < 0.8, lambda: glh.qz_prev, lambda: qz)

    new_grounding = eqx.tree_at(lambda x: x.qz_prev, glh, qz)
    return new_grounding, ells[qz.argmax()], qz
