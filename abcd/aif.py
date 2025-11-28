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


from itertools import product

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from pymdp.maths import compute_accuracy

from pymdp.maths import multidimensional_outer, spm_wnorm, factor_dot


def generate_policies(time_horizon, n_actions, n_modalities):
    policies = list(product(jnp.arange(n_actions).tolist(), repeat=time_horizon))
    policies = jnp.array(policies)[..., None].repeat(n_modalities, -1)
    return policies


@jax.jit
def compute_param_infogain(pA_m, qo_m, qs_deps):
    wa_m = spm_wnorm(pA_m) * (pA_m > 0.0)
    fd = factor_dot(wa_m, qs_deps, keep_dims=(0,))[..., None]
    return qo_m.dot(fd)


def free_energy_jax(qs, prior):
    # Neg-entropy of posterior marginal H(q[f])
    negH_qs = qs.dot(jnp.log(qs[:, np.newaxis] + 1e-16))
    # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
    xH_qp = -qs.dot(prior[:, np.newaxis])
    free_energy = negH_qs + xH_qp

    return free_energy, None


def free_energy(qs, prior, likelihood=None):
    free_energy = 0
    # Neg-entropy of posterior marginal H(q[f])
    negH_qs = qs.dot(np.log(qs[:, np.newaxis] + 1e-16))
    # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
    xH_qp = -qs.dot(prior[:, np.newaxis])
    free_energy += negH_qs + xH_qp

    acc = None
    if likelihood is not None:
        acc = compute_accuracy(likelihood, qs, likelihood)
        free_energy -= acc
    return free_energy, acc


@jax.vmap
def count(*args):
    """
    Double vmapped counting method, i.e. args and a have to be of shape
    (n_batch, n_time, state_dim)

    can be arbitrary many: [state_i, state_dep..., action]
     but for this case it would be [s1, s0, a]

    """
    return jax.vmap(multidimensional_outer)(args)


def update_B(qB, cats, actions, B_dependencies, lr_pB=1.0):
    a = jax.nn.one_hot(actions, num_classes=4)
    # split in previous (t=0) states (s0) and current (t=1) states (s1)
    # with a being the action at t=0 to go from s0 -> s1

    s0 = jtu.tree_map(lambda x: x[:, :-1], cats)
    s1 = jtu.tree_map(lambda x: x[:, 1:], cats)

    for f in range(len(qB)):
        deps = [s0[fi] for fi in B_dependencies[f]]
        dfdb = count(s1[f], *deps, a).sum(axis=[0, 1])
        qB[f] = qB[f] + lr_pB * jnp.expand_dims(dfdb, 0)

    E_qB = [b / b.sum(axis=1, keepdims=True) for b in qB]
    return qB, E_qB
