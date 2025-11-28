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

import pickle
import copy

import rich

from pathlib import Path

from tqdm import trange

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from pymdp.agent import Agent as PyMDPAgent

import abcd
from abcd.agent.clone_agent import CloneAgent, Buffer
from abcd.agent.hai_agent import HierarchicalAgent
from abcd.utils import lr2obs
from abcd import maze

from argparse import ArgumentParser


def load_models(load_path, navigation_path, task_path):
    with open(load_path / navigation_path, "rb") as f:
        navigation = pickle.load(f)
    with open(load_path / task_path, "rb") as f:
        task = pickle.load(f)
    return navigation, task


def env_step_fn(carry, t):
    agent, env, key, observation_t, reward_t = carry

    state_pref_t, location_t = agent.low.pymdp_agent.H[0], env.agent_location

    key, subkey = jr.split(key)
    agent, action_t, agent_info = agent.act(observation_t, reward_t, subkey)

    (observation_next, reward_next), env = env.act(action_t)

    info = {
        "observation": observation_t,
        "reward": reward_t,
        "action": action_t,
        "location": location_t,
        "state_preference": state_pref_t,
    }
    info.update(agent_info)

    return (agent, env, key, observation_next, reward_next), info


def get_shai_agent(
    load_path,
    remap,
    mixture,
    random_high,
    learn_schema=False,
    task_model=None,
    env_prefix="large_maze",
):
    if task_model is None:
        task_model = "schema_1_1_clone.pickle"

    navigation_cscg, schema_cscg = load_models(
        load_path, f"{env_prefix}_navigation.pickle", task_model
    )

    navigation, nav_active_states = CloneAgent.from_cscg(
        navigation_cscg,
        death_state=False,
        horizon_length=1,
        reduce=True,
        inductive=True,
    )

    task_active = jnp.array(
        sorted(
            jax.vmap(lambda x: lr2obs(x, 0))(nav_active_states).tolist()
            + jax.vmap(lambda x: lr2obs(x, 1))(nav_active_states).tolist()
        )
    )

    # Take the structure from the cscg
    n_clones = schema_cscg.n_clones[0]
    use_smoothing = bool(n_clones > 1 and learn_schema)
    schema, _ = CloneAgent.from_cscg(
        schema_cscg,
        death_state=False,
        horizon_length=3,
        reduce=True,
        active_obs=task_active,
        inductive=True,
        sort_states=True,
        use_smoothing=use_smoothing,
    )

    if learn_schema:
        # If learn schema, we assume identity likelihood
        n_high_obs = schema.pymdp_agent.A[0].shape[1]
        new_A = jnp.zeros((n_high_obs, n_clones * n_high_obs))
        for i in range(n_clones):
            new_A = new_A.at[:, jnp.arange(i, n_high_obs * n_clones, n_clones)].set(
                jnp.eye(n_high_obs)
            )
        new_A = new_A[None]

        new_B = jr.uniform(
            jr.PRNGKey(0), (1, n_high_obs * n_clones, n_high_obs * n_clones, 1)
        )
        if use_smoothing:
            new_B = new_B * 0.075

        schema_pydmp_agent = PyMDPAgent(
            A=[new_A],
            B=[new_B / new_B.sum(1)],
            pB=[new_B],
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            learn_A=False,
            learn_B=True,
            action_selection="stochastic",
            gamma=1,
            policy_len=3,
            policies=schema.pymdp_agent.policies,
            onehot_obs=False,
            apply_batch=False,
            use_param_info_gain=True,
        )
        schema = eqx.tree_at(
            lambda x: (x.pymdp_agent, x.empirical_prior),
            schema,
            (schema_pydmp_agent, schema_pydmp_agent.D),
        )

        obs_dim, state_dim = new_A[0].shape
        buffer = Buffer(
            10, state_dim_size=state_dim, obs_dim_size=obs_dim, action_dim_size=1
        )
        buffer = buffer.add(schema_pydmp_agent.D[0].flatten(), 0, 0)
        schema = eqx.tree_at(lambda x: x.buffer, schema, buffer)

    ag = HierarchicalAgent(
        low=navigation,
        high=schema,
        remap=remap,
        use_mixture=mixture,
        random_high=random_high,
        ell_treshold=-5.0,
        sample_preference=False,
        gl_gamma=1.0,
    )

    gp = copy.deepcopy(ag.grounding.prior)
    lo_ep = copy.deepcopy(ag.low.empirical_prior)
    hi_ep = copy.deepcopy(ag.high.empirical_prior)

    return ag, (gp, lo_ep, hi_ep)


def run_simulation(
    load_path,
    store_path,
    n_environments,
    mixture,
    random_high=False,
    remap=True,
    learn_schema=False,
    task_model=None,
    n_steps=3000,
    alternation=False,
    repeat_envs=False,
    n_envs_to_cycle=20,
    verbose=False,
    large_maze=True,
):
    store_path.mkdir(exist_ok=True, parents=True)

    # Generate the agent for this simulation
    agent, (gp, lo_ep, hi_ep) = get_shai_agent(
        load_path,
        remap=remap,
        mixture=mixture,
        random_high=random_high,
        learn_schema=learn_schema,
        task_model=task_model,
        env_prefix="large_maze" if large_maze else "small_maze",
    )

    rewards_list = []
    key = jr.PRNGKey(42)  # A key for cycling the repeat envs.
    for episode_index in trange(n_environments):
        # Reset the agent.
        low = eqx.tree_at(lambda x: x.empirical_prior, agent.low, copy.deepcopy(lo_ep))
        high = eqx.tree_at(
            lambda x: x.empirical_prior, agent.high, copy.deepcopy(hi_ep)
        )

        # Uniform over the used clusters
        uni_qz = agent.grounding.used_mask / agent.grounding.used_mask.sum()
        grounding = eqx.tree_at(lambda x: (x.qz_prev), agent.grounding, uni_qz)

        # basically if we are using a mixture
        if agent.grounding.prior.shape[0] == 1:
            grounding = eqx.tree_at(lambda x: x.prior, grounding, gp)

        agent = eqx.tree_at(lambda x: x.low, agent, low)
        agent = eqx.tree_at(lambda x: x.high, agent, high)
        agent = eqx.tree_at(lambda x: x.high_qs, agent, high.pymdp_agent.D)
        agent = eqx.tree_at(lambda x: x.grounding, agent, grounding)

        if repeat_envs:
            key, subkey = jr.split(key)
            env_key = jr.choice(subkey, jnp.arange(n_envs_to_cycle))
        else:
            env_key = episode_index

        # Reset the environment
        env = maze.MazeEnvironment(
            nrows=3,
            ncols=3,
            corridor_size=1,
            outer_wall=True,
            key=jr.PRNGKey(env_key),
            alternation=alternation,
            large_maze=large_maze,
        )
        (obs_0, rew_0), env = env.reset(jr.PRNGKey(env_key))

        (agent, *_), info = jax.lax.scan(
            env_step_fn, (agent, env, jr.PRNGKey(0), obs_0, rew_0), jnp.arange(n_steps)
        )

        rewards_list.append(int(info["reward"].sum()))
        if verbose:
            print(
                f"[ENV: {env_key}]: Rewards: {rewards_list[-1]} min(ell): {info['max_ell'].min():.3f} # Clusters: {agent.grounding.used_mask.sum()}"
            )

        info.update({"grounding_prior": agent.grounding.prior})
        info.update({"schema_B": agent.high.pymdp_agent.B[0]})
        info.update({"schema_A": agent.high.pymdp_agent.A[0]})
        if agent.high.pymdp_agent.learn_B:
            info.update({"schema_pB": agent.high.pymdp_agent.pB[0]})
        np.savez_compressed(
            store_path / f"episode_{episode_index}",
            **{k: np.array(v) for k, v in info.items()},
        )

    print(f"Ran experiment: {store_path.stem.replace('_', ' ')}")
    print(f"And stored at {store_path.parent}")
    print(f"Collected rewards: {rewards_list}")
    return None


def cfg_to_string(cfg):
    name = ""
    for k, v in cfg.items():
        val = str(v).lower().replace(".pickle", "").split("/")[-1]
        name += f"_{str(k).lower().replace('_', '')}:{val}"
    return name[1:]


def get_configs(env_name):
    if "abcd" in env_name:
        configs = [
            {  # Random agent
                "mixture": False,
                "random_high": True,
                "remap": False,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI L
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": True,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI K
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI L MoG
                "mixture": True,
                "random_high": False,
                "remap": True,
                "learn_schema": True,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI K MoG
                "mixture": True,
                "random_high": False,
                "remap": True,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # HAI-2
                "mixture": False,
                "random_high": False,
                "remap": False,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_2_4_clone.pickle",
            },
            {  # HAI-20
                "mixture": False,
                "random_high": False,
                "remap": False,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_20_40_clone.pickle",
            },
            {  # HAI-40
                "mixture": False,
                "random_high": False,
                "remap": False,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_40_80_clone.pickle",
            },
            # {  # HAI-1
            #     "mixture": False,
            #     "random_high": False,
            #     "remap": False,
            #     "learn_schema": False,
            #     "task_model": f"{env_name}/schema_1_1_clone.pickle",
            # },
        ]
    elif "abcb" in env_name:
        configs = [
            {  # S-HAI L
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": True,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI K
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
            {  # S-HAI-2C L
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": True,
                "task_model": f"{env_name}/schema_1_2_clone.pickle",
            },
            {  # S-HAI-2C L
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_2_clone.pickle",
            },
            {  # Random agent
                "mixture": False,
                "random_high": True,
                "remap": False,
                "learn_schema": False,
                "task_model": f"{env_name}/schema_1_1_clone.pickle",
            },
        ]

    return configs


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--alternation", action="store_true")
    ap.add_argument("--large_maze", action="store_true")
    ap.add_argument("--repeat_envs", action="store_true")

    ap.add_argument("--load_from", type=str, default="data/learned-models-offline")
    ap.add_argument("--store_at", type=str, default="data/maze-simulations")
    ap.add_argument("--n_envs", type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=10_000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Generic experiment setup
    root_path = Path(abcd.__file__).parent.parent
    load_path = root_path / args.load_from
    if not load_path.exists():
        raise FileNotFoundError(f"{load_path} not found. Please model training first")

    store_path = root_path / args.store_at
    store_path.mkdir(parents=True, exist_ok=True)

    suffix = "abcb" if args.alternation else "abcd"
    prefix = "large" if args.large_maze else "small"
    sufsuffix = "_repeats" if args.repeat_envs else ""

    model_env_name = f"{prefix}_maze_{suffix}"
    env_name = model_env_name + sufsuffix

    configs = get_configs(model_env_name)

    if args.debug:
        # When debugging, just fill in this config, with the particular model you want
        # to debug.
        configs = [
            {  # S-HAI-L
                "mixture": False,
                "random_high": False,
                "remap": True,
                "learn_schema": True,
                "task_model": f"{env_name}/schema_1_2_clone.pickle",
            }
        ]

    for config in configs:
        rich.print(config)

        experiment_store_path = store_path / env_name / cfg_to_string(config)
        run_simulation(
            load_path,
            experiment_store_path,
            n_environments=args.n_envs,
            n_steps=args.n_steps,
            alternation=args.alternation,
            repeat_envs=args.repeat_envs,
            verbose=args.verbose,
            large_maze=args.large_maze,
            **config,
        )
