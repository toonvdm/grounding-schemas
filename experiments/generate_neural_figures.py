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
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import matplotlib.patheffects as pe

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as nn

from pymdp import control

import abcd
from abcd import maze

from run_maze_experiment import get_shai_agent


def get_env_info(i=0, large_maze=False):
    env = maze.MazeEnvironment(
        nrows=3,
        ncols=3,
        corridor_size=1,
        outer_wall=True,
        key=jr.PRNGKey(i),
        alternation=False,
        large_maze=large_maze,
    )
    _, env = env.reset(jr.PRNGKey(i))

    return env.layout, env.reward_options[env.reward_sequence], ["A", "B", "C", "D"]


def plot_env(ax, layout, rlocs, rlabels, grayscale=False, alpha=0.15):
    if grayscale:
        colors = ["#ffffff", "#000000", "#000000", "#000000", "#000000", "#000000"]
    else:
        colors = ["#ffffff", "#edae49", "#d1495b", "#00798c", "#003d5b", "#218655"]

    my_cmap = ListedColormap(colors, name="my_cmap")

    if len(jnp.unique(layout)) > len(colors):
        my_cmap = plt.get_cmap("Greens")

    ax.imshow(layout, cmap=my_cmap, alpha=alpha)
    for rloc, rlabel in zip(rlocs, rlabels):
        t = ax.text(
            rloc[0],
            rloc[1],
            rlabel,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=5,
            fontweight="bold",
        )
        t.set_path_effects([pe.withStroke(linewidth=0.5, foreground="w", alpha=0.85)])


def create_envs_plot(
    filename, grayscale=False, alpha=0.45, large_maze=False, show=False
):
    fig, ax = plt.subplots(1, 4, figsize=(4, 1.4), dpi=300)
    plt.suptitle("ABCD Task", fontsize=10, y=0.925)
    [
        plot_env(
            a, *get_env_info(i, large_maze=large_maze), grayscale=grayscale, alpha=alpha
        )
        for i, a in enumerate(ax)
    ]
    [ax.flatten()[i].set_title(f"Block {i + 1}", fontsize=8) for i in range(4)]
    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]

    plt.subplots_adjust(bottom=0.0, top=0.7, left=0.05, right=0.95)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def create_activations_plot(
    filename,
    states,
    locations,
    n,
    std=0.1,
    cmap=None,
    show=False,
    large_maze=False,
    title=None,
):
    fig, ax = plt.subplots(1, 4, figsize=(4, 1.4), dpi=300)
    key, *subkeys = jr.split(jr.PRNGKey(0), 6)
    [
        plot_env(a, *get_env_info(i, large_maze=large_maze), grayscale=large_maze)
        for i, a in enumerate(ax)
    ]
    [ax.flatten()[i].set_title(f"Block {i + 1}", fontsize=8) for i in range(4)]

    if title is not None:
        plt.suptitle(title, fontsize=10, y=0.925)

    num_colors = int(states[0][0, :, 0, 0].argmax(-1).max() + 2)
    if cmap is None:
        if num_colors > 20:
            hues = jr.uniform(subkeys[0], shape=(num_colors))
            # Fixed saturation and value for vivid colors
            saturations = (
                np.array(jr.uniform(subkeys[2], shape=(num_colors))) * 0.85 + 0.15
            )
            values = np.array(jr.uniform(subkeys[3], shape=(num_colors))) * 0.75 + 0.25

            # s = 0.65, v = 0.9 if fixed
            colors = hsv_to_rgb(np.stack([hues, saturations, values], axis=1))
            cmap = ListedColormap(colors)

        else:
            cmap = "tab20c"

    for i in range(4):
        loc = locations[i]
        x, y = loc[n:, 0], loc[n:, 1]

        # Add some noise to not have overlapping visual
        xn = std * jr.normal(subkeys[0], shape=x.shape)
        yn = std * jr.normal(subkeys[1], shape=y.shape)

        st = states[i][0, n:, 0, 0].argmax(-1)
        ax[i].scatter(
            x + xn, y + yn, marker=".", s=1, c=st, cmap=cmap, vmin=0, vmax=num_colors
        )

    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]

    plt.suptitle(title, fontsize=10, y=0.925)
    # plt.subplots_adjust(top=0.7, left=0.05, right=0.95)
    plt.subplots_adjust(bottom=0.0, top=0.7, left=0.05, right=0.95)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def joint_low_high(low_state, high_state, reward, locations):
    """
    Creates joint values for the low and high states, and separates out only rewarding
    locations (reduces to unique indices to reduce the amount of colors we need to map)
    """
    idcs = [jnp.where(r)[0] for r in reward]
    y = np.array(
        [
            (high_state[i].argmax(-1) + 1) * (1 + low_state[i].argmax(-1))
            for i in range(5)
        ]
    )

    states = [y[i][:, j] for i, j in enumerate(idcs)]
    locs = [locations[i][j] for i, j in enumerate(idcs)]

    uni_y = jnp.unique(jnp.concatenate([jnp.unique(yi) for yi in states]))

    _map = jnp.zeros(y.max()).at[uni_y].set(jnp.arange(len(uni_y)) + 1)

    y = _map[y].astype(jnp.int32)
    y = nn.one_hot(y, y.max(), axis=-1)
    y = [y[i][:, j] for i, j in enumerate(idcs)]

    return y, locs


def get_progress_values(H_low, low_state, locations, large_maze=False):
    # Basically construct an agent to get to these values (B matrix & policies list).
    prefix = "large_maze" if large_maze else "small_maze"
    config = {
        "mixture": False,
        "random_high": False,
        "remap": True,
        "learn_schema": False,
        "task_model": f"{prefix}_abcd/schema_1_1_clone.pickle",
    }
    # Generate the agent for this simulation
    root_path = Path(abcd.__file__).parent.parent
    agent, _ = get_shai_agent(
        root_path / "data/learned-models-offline", env_prefix=prefix, **config
    )

    mat_I = []
    for i in range(5):
        mat_H = H_low[i]
        mat_I.append(
            jax.vmap(
                lambda mat_Hi: jax.vmap(partial(control.generate_I_matrix, depth=40))(
                    [mat_Hi], agent.low.pymdp_agent.B, jnp.array([0.1])
                )
            )(mat_H)[0]
        )

    Eqpi_I = [
        ((low_state[i][0, ..., 0, :] * mat_I[i].sum(-2))[None, :, None]).sum(-1)
        for i in range(5)
    ]

    prefs = [H_low[i].argmax(-1)[-1000:, 0] for i in range(5)]
    vals = [
        Eqpi_I[i][0, -1000:, 0, 0] / Eqpi_I[i][0, -1000:, 0, 0].max() for i in range(5)
    ]
    locs = [locations[i][-1000:] for i in range(5)]
    return prefs, vals, locs


def create_progress_circle_plots(
    filename, low_state, H_low, locations, show=False, large_maze=False
):
    prefs, vals, locs = get_progress_values(
        H_low, low_state, locations, large_maze=large_maze
    )
    rews = [rewards[i][-1000:] for i in range(5)]

    fig, ax = plt.subplots(1, 4, figsize=(4, 1.4), dpi=300)

    plt.suptitle("Goal-Progress Activity", fontsize=10, y=0.925)

    stage_cmap = plt.get_cmap("tab20", 4)
    for i in range(4):
        env_info = get_env_info(i, large_maze=large_maze)

        # Get the last succesful reward sequence
        idcs = jnp.where(rews[i] == 1)[0][-6:-1]
        start, stop = idcs[0], idcs[-1] + 1

        r, theta = 0.4, jnp.linspace(2 * jnp.pi, 0, stop - start)
        x, y = r * jnp.cos(theta), r * jnp.sin(theta)

        ax[i].scatter(x, y, marker="o", c=vals[i][start:stop], s=10)
        ax[i].set_aspect("equal")

        for j, pref in enumerate(prefs[i][start:stop]):
            theta_inter = jnp.linspace(theta[j], theta[j + 1], 100)
            xi, yi = r * jnp.cos(theta_inter), r * jnp.sin(theta_inter)
            ax[i].plot(xi, yi, c="lightgray", zorder=-2)

        ax[i].scatter([0], [0], marker="o", color="lightgray", s=15)
        for j, rew in enumerate(rews[i][start:stop]):
            if rew:
                # Detect whether it is A B C or D
                current_loc = jnp.array(locs[i][start:stop][j])
                goal_idx = jnp.where(
                    jnp.array([jnp.all(current_loc == e) for e in env_info[1]])
                )[0][0]

                ax[i].plot(
                    [0, x[j]],
                    [0, y[j]],
                    color=stage_cmap(goal_idx),
                    zorder=-1,
                    alpha=1,
                    linestyle="-",
                )

                xi, yi = x[j], y[j]
                xt = xi + np.cos(np.arctan2(yi, xi)) * 0.14
                yt = yi + np.sin(np.arctan2(yi, xi)) * 0.14

                ax[i].text(
                    xt,
                    yt,
                    env_info[2][goal_idx],
                    fontsize=6,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )

            v = 0.6
            ax[i].set_xlim([-v, v])
            ax[i].set_ylim([-v, v])

        ax[i].set_title(f"Block {i + 1}", fontsize=8)

    [a.axis("off") for a in ax.flatten()]

    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    h, w = 0.35, 0.0125
    h0 = (0.7 - h) / 2
    cax = fig.add_axes([0.94, h0, w, h])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.tick_params(labelsize=7)

    fig.text(0.94 + w / 2, h0 - 0.025, "Low", va="top", ha="center", fontsize=6)
    fig.text(
        0.94 + w / 2,
        h0 + h + 0.025,
        "High",
        va="bottom",
        ha="center",
        fontsize=6,
    )

    # plt.subplots_adjust(top=0.7, left=0.05, right=0.95)
    plt.subplots_adjust(bottom=0.0, top=0.7, left=0.01, right=0.90)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def create_progress_activation_plot(
    filename,
    low_state,
    H_low,
    locations,
    std=0.1,
    show=False,
    large_maze=False,
    title=None,
):
    prefs, vals, locs = get_progress_values(
        H_low, low_state, locations, large_maze=large_maze
    )
    prefs, vals, locs = prefs[0], vals[0], locs[0]

    key, *subkeys = jr.split(jr.PRNGKey(0), 3)

    fig, ax = plt.subplots(1, 4, figsize=(4, 1.4), dpi=300)

    if title is not None:
        plt.suptitle(title, fontsize=10, y=0.925)
    key, *subkeys = jr.split(jr.PRNGKey(0), 3)
    [
        plot_env(a, *get_env_info(0, large_maze=large_maze), grayscale=True)
        for i, a in enumerate(ax)
    ]
    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]

    unique_prefs = jnp.unique(prefs)

    titles = [
        r"$A \rightarrow B$",
        r"$B \rightarrow C$",
        r"$C \rightarrow D$",
        r"$D \rightarrow A$",
    ]
    for i in range(4):
        ax.flatten()[i].set_title(titles[i], fontsize=8)
        _filter = prefs == unique_prefs[i]
        x, y = locs[_filter, 0], locs[_filter, 1]

        # Add some noise to not have overlapping visual
        xn = std * jr.normal(subkeys[0], shape=x.shape)
        yn = std * jr.normal(subkeys[1], shape=y.shape)

        ax.flatten()[i].scatter(
            x + xn, y + yn, c=vals[_filter], cmap="viridis", marker=".", s=0.35
        )

    # plt.subplots_adjust(top=0.70, left=0.05, right=0.95)
    plt.subplots_adjust(bottom=0.0, top=0.7, left=0.05, right=0.95)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    root_path = Path(abcd.__file__).parent.parent

    ap = ArgumentParser()
    ap.add_argument("--load_from", type=str, default="data/maze-simulations")
    ap.add_argument("--store_at", type=str, default="data/paper_figures")
    ap.add_argument("--alternation", action="store_true")
    ap.add_argument("--repeat_envs", action="store_true")
    ap.add_argument("--large_maze", action="store_true")
    args = ap.parse_args()

    prefix = "large" if args.large_maze else "small"
    task = "abcb" if args.alternation else "abcd"
    sufsuffix = "_repeats" if args.repeat_envs else ""
    env_name = f"{prefix}_maze_{task}" + sufsuffix
    sim_path = root_path / args.load_from / env_name

    store_path = root_path / args.store_at / env_name / "neural"
    store_path.mkdir(exist_ok=True, parents=True)

    # S-HAI K
    tag = "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone"

    # Load the variables we are interested in for the first 5 envs
    locations, high_state, low_state, low_H, rewards = [], [], [], [], []
    for i in range(5):
        d = dict(np.load(sim_path / tag / f"episode_{i}.npz"))
        locations.append(d["location"])
        high_state.append(d["high_state"])
        low_state.append(d["low_state"])
        rewards.append(d["reward"])
        low_H.append(d["state_preference"])

    create_progress_circle_plots(
        store_path / "progress_circles.pdf",
        low_state,
        low_H,
        locations,
        large_maze=args.large_maze,
    )

    create_envs_plot(store_path / "envs.pdf", large_maze=args.large_maze, alpha=0.75)
    create_activations_plot(
        store_path / "high_activations.pdf",
        high_state,
        locations,
        n=-1000,
        large_maze=args.large_maze,
        title="Level 2 State Activations",
    )
    create_activations_plot(
        store_path / "low_activations.pdf",
        low_state,
        locations,
        n=-1000,
        large_maze=args.large_maze,
        title="Level 1 State Activations",
    )
    create_activations_plot(
        store_path / "remap_activations.pdf",
        *joint_low_high(low_state, high_state, rewards, locations),
        n=-1000,
        std=0.25 if args.large_maze else 0.1,
        cmap=ListedColormap(
            [plt.get_cmap("Set1")(i) for i in range(8)]
            + [plt.get_cmap("Set3")(i) for i in range(9)]
        ),
        large_maze=args.large_maze,
        title="Goal-and-Location Activations",
    )

    create_progress_activation_plot(
        store_path / "progress_activations.pdf",
        low_state,
        low_H,
        locations,
        large_maze=args.large_maze,
        title="Simulated Progress Cells",
    )
