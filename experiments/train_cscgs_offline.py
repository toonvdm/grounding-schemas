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

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx

import abcd
from abcd import maze
from abcd import cscg
from abcd.visualize import plot_graph
from abcd.utils import lr2obs

from argparse import ArgumentParser


def decode(chmm, episode):
    obs = np.array(episode[:, 0]).astype(np.int64)
    action = np.array(episode[:, 2]).astype(np.int64)
    _, states = chmm.decode(obs, action)
    return states


def fit_cscg(states, action, n_clones=1, n_obs=None, n_steps=1000):
    assert states.shape == action.shape
    if n_obs is None:
        n_obs = states.max() + 1
    n_clones = n_clones * np.ones(n_obs).astype(np.int64)
    chmm = cscg.CHMM(n_clones=n_clones, x=states, a=action, pseudocount=2e-3)

    convs = []
    conv = chmm.learn_em_T(states, action, n_iter=n_steps, term_early=True)
    chmm.pseudocount = 0.0
    conv = chmm.learn_viterbi_T(states, action, n_iter=100)
    convs.append(conv)

    return np.concatenate(convs, 0), chmm


def fit_task_cscg(states, action, n_clones_int=1, n_obs=None, n_steps=1000):
    assert states.shape == action.shape
    if n_obs is None:
        n_obs = states.max() + 1

    _inv_map = np.unique(states).astype(np.int64)
    _map = np.zeros(n_obs)
    _map[_inv_map] = np.arange(len(_inv_map)).astype(np.int64)

    states = _map[states].astype(np.int64)

    n_obs_new = len(_inv_map)

    n_clones = n_clones_int * np.ones(n_obs_new).astype(np.int64)
    chmm = cscg.CHMM(n_clones=n_clones, x=states, a=action, pseudocount=1e-5)

    convs = []
    conv = chmm.learn_em_T(states, action, n_iter=n_steps, term_early=True)
    chmm.pseudocount = 0.0
    conv = chmm.learn_viterbi_T(states, action, n_iter=100)
    convs.append(conv)

    n_clones = n_clones_int * np.ones(n_obs).astype(np.int64)

    chmm._inv_map = _inv_map
    chmm._map = _map

    return np.concatenate(convs, 0), chmm


def generate_data(
    n_steps,
    env_seed=0,
    switch_env_every=None,
    n_envs_to_learn=1,
    alternation=False,
    large_maze=True,
):
    if switch_env_every is None:
        switch_env_every = n_steps + 1

    def step_fn(carry, t):
        env, a_t, key, env_seed = carry
        key, *subkeys = jr.split(key, 4)

        o_tn, env = env.act(a_t)

        # Reset the environment, and restore player position
        agent_loc = env.agent_location
        new_seed = (env_seed + 1) % n_envs_to_learn
        # new_seed = jr.choice(subkeys[0], jnp.arange(n_envs_to_learn))
        env, env_seed = jax.lax.cond(
            (t % switch_env_every) == 0,
            lambda: (env.reset(jr.PRNGKey(new_seed))[1], new_seed),
            lambda: (env, env_seed),
        )
        env = eqx.tree_at(lambda x: x.agent_location, env, agent_loc)
        ####

        a_tn = jr.choice(subkeys[2], jnp.arange(4))

        return (env, a_tn, key, env_seed), (env, o_tn, a_tn, env_seed)

    env = maze.MazeEnvironment(
        nrows=3,
        ncols=3,
        corridor_size=1,
        outer_wall=True,
        key=jr.PRNGKey(env_seed),
        alternation=alternation,
        large_maze=large_maze,
    )
    _, env = env.reset(jr.PRNGKey(env_seed))

    _, (envs, obs_rew, actions, env_seeds) = jax.lax.scan(
        step_fn, (env, jnp.array(0), jr.PRNGKey(0), -1), jnp.arange(n_steps)
    )

    obs = obs_rew[:, 0]
    rewards = obs_rew[:, 1]

    return obs, rewards, actions, env_seeds


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--load_navigation", action="store_true")
    ap.add_argument("--n_nav_clones", type=int, default=100)
    ap.add_argument("--n_task_clones", type=int, default=1)
    ap.add_argument("--large_maze", action="store_true")
    ap.add_argument("--alternation", action="store_true")
    args = ap.parse_args()

    env_tag = "large_maze" if args.large_maze else "small_maze"

    store_path = Path(abcd.__file__).parent.parent / "data/learned-models-offline"
    store_path.mkdir(exist_ok=True, parents=True)

    o, r, a, s = generate_data(50_000, large_maze=args.large_maze)

    # cast the data to numpy array
    o = np.array(o).astype(np.int32)
    a = np.array(a).astype(np.int32)
    r = np.array(r).astype(np.int32)

    # Fit the navigation cscg
    if args.load_navigation:
        with open(store_path / f"{env_tag}_navigation.pickle", "rb") as f:
            chmm_orig = pickle.load(f)
    else:
        _, chmm_orig = fit_cscg(o, a, args.n_nav_clones, n_steps=1000)
        with open(store_path / f"{env_tag}_navigation.pickle", "wb") as f:
            pickle.dump(chmm_orig, f)

    n_obs = len(np.unique(o))
    plot_graph(
        chmm_orig,
        o,
        a,
        store_path / f"{env_tag}_navigation_graph.pdf",
        n_obs + 1,
        n_clones=args.n_nav_clones,
        cmap=plt.get_cmap("tab20c"),
    )

    print(f"Running for alternation: {args.alternation}")
    store_path_i = store_path / (
        f"{env_tag}_abcb" if args.alternation else f"{env_tag}_abcd"
    )
    store_path_i.mkdir(exist_ok=True, parents=True)
    for n_envs in [1, 1_2, 2, 20, 40]:
        n_task_clones = n_envs * 2

        # Because for alternation we _do_ want to have 2 clones for our first env
        if n_envs == 1:
            n_envs, n_task_clones = 1, 1
        elif n_envs == 1_2:
            n_envs, n_task_clones = 1, 2

        print(f"\n\t{n_envs} environments: {n_task_clones} task clones...")

        switch_every = 1000 * (10 if args.large_maze else 1)
        n = n_envs * switch_every
        o, r, a, s = generate_data(
            n,
            switch_env_every=switch_every,
            n_envs_to_learn=n_envs,
            alternation=args.alternation,
            large_maze=args.large_maze,
        )

        # cast the data to numpy array
        o = np.array(o).astype(np.int32)
        a = np.array(a).astype(np.int32)
        r = np.array(r).astype(np.int32)

        print("\tReward per train episode: ", r.reshape(-1, switch_every).sum(-1))

        episodes = jnp.concatenate([o[..., None], r[..., None], a[..., None]], axis=-1)

        nav_states = decode(chmm_orig, episodes)

        # Filter out timesteps where reward was observed
        episode_hl = episodes[r == 1]
        nav_states = nav_states[r == 1]

        # Set the action to 0 for the high level
        episode_hl = episode_hl.at[:, 2].set(0)

        # Map the low-level state & reward to high-level observations
        obs_hl = np.array(jax.vmap(lr2obs)(nav_states, episode_hl[:, 1]))
        # At the higher level, the only action is "step through the cycle"
        act_hl = np.array(0 * obs_hl)

        _, task = fit_task_cscg(
            obs_hl, act_hl, n_obs=1000, n_clones_int=n_task_clones, n_steps=2000
        )
        with open(
            store_path_i / f"schema_{n_envs}_{n_task_clones}_clone.pickle", "wb"
        ) as f:
            pickle.dump(task, f)

        plot_graph(
            task,
            task._map[obs_hl].astype(np.int64),
            act_hl,
            store_path_i / f"task_graph_{n_envs}_{n_task_clones}.pdf",
            n_obs * args.n_nav_clones * 2,
            n_clones=n_task_clones,
            cmap=plt.get_cmap("tab20c"),
        )
