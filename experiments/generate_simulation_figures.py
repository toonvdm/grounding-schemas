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

import abcd
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm

from scipy.stats import ttest_rel

import seaborn as sns

import networkx as nx
from abcd import maze
import jax.random as jr

from argparse import ArgumentParser


def create_plot(
    mode="square", xlabel=None, ylabel=None, figsize=None, set_box_aspect=True
):
    if mode == "square":
        if figsize is None:
            figsize = (2, 2)
        box_aspect = 1
        left = 0.3
        right = 0.95
    elif mode == "wide":
        if figsize is None:
            figsize = (4.5, 2)
        box_aspect = 0.5
        left = 0.15
        right = 0.915

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    # ax.spines[["top", "right"]].set_visible(False)
    if set_box_aspect:
        ax.set_box_aspect(box_aspect)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    plt.subplots_adjust(bottom=0.25, left=left, right=right)
    return fig, ax


def layout_to_nx(grid):
    graph = nx.grid_2d_graph(*grid.shape)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == -1:
                graph.remove_node((r, c))
    return graph


def get_optimal_reward_rate_abcd(alternation=False, repeat_envs=False, large_maze=True):
    """
    For the maze environment, compute the average reward per step, if you would take a
    perfect trajectory
    """
    avg_reward_per_step = []
    optimal_steps_per_trial = []

    key = jr.PRNGKey(42)  # Env seed key
    for i in range(40):
        if repeat_envs:
            key, subkey = jr.split(key)
            env_key = jr.choice(subkey, jnp.arange(20))
        else:
            env_key = i

        env = maze.MazeEnvironment(
            nrows=3,
            ncols=3,
            corridor_size=1,
            outer_wall=True,
            key=jr.PRNGKey(env_key),
            alternation=alternation,
            large_maze=large_maze,
        )
        graph = layout_to_nx(env.layout)

        a, b, c, d = env.reward_options[env.reward_sequence]

        ab = nx.shortest_path(graph, source=tuple(a.tolist()), target=tuple(b.tolist()))
        bc = nx.shortest_path(graph, source=tuple(b.tolist()), target=tuple(c.tolist()))
        cd = nx.shortest_path(graph, source=tuple(c.tolist()), target=tuple(d.tolist()))
        da = nx.shortest_path(graph, source=tuple(d.tolist()), target=tuple(a.tolist()))

        # 4 rewards acquired, - the start location step of getting reward
        avg_reward_per_step += [4 / (len(ab) + len(bc) + len(cd) + len(da) - 4)]
        optimal_steps_per_trial += [(len(ab) + len(bc) + len(cd) + len(da) - 4)]

    return jnp.array(optimal_steps_per_trial), jnp.array(avg_reward_per_step).mean()


def get_rewards(experiment_path, n=250):
    files = list(experiment_path.glob("*"))
    files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))
    raw_rewards, convolved_rewards = [], []
    for i in files:
        rewards = dict(np.load(i, allow_pickle=True))["reward"]
        raw_rewards.append(rewards)

        rewards = np.concatenate([np.zeros((n - 1)), rewards], 0)
        rewards_conv = jnp.convolve(rewards, np.ones(n) / n, mode="valid")
        convolved_rewards.append(rewards_conv)

    convolved_rewards = np.array(convolved_rewards)
    raw_rewards = np.array(raw_rewards)

    return raw_rewards, convolved_rewards


def get_locations(experiment_path):
    files = list(experiment_path.glob("*"))
    files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))
    all_locations = []
    for i in files:
        all_locations.append(dict(np.load(i, allow_pickle=True))["location"])
    return all_locations


def get_schema_likelihoods(experiment_path):
    files = list(experiment_path.glob("*"))
    files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))
    all_lhs = []
    for i in files:
        all_lhs.append(dict(np.load(i, allow_pickle=True))["schema_A"])
    return all_lhs


def get_num_steps_to_first(r):
    # If the steps to first is not found, it will be filled in with the max step value
    steps_to_first = jnp.ones(r.shape[0]) * r.shape[1]
    res = jnp.where(r * r.cumsum(-1) == 5)
    if len(res[0]) == 0:
        a, b = res
        steps_to_first = steps_to_first.at[a].set(b)
    return steps_to_first


def get_steps_per_trial(r, optimal=None, trial_length=4):
    if optimal is None:
        # if optimal is set, the returned steps will be the ratio of steps / optimal
        optimal = jnp.ones(r.shape[0])

    all_steps = []
    for i, (r_i, o_i) in enumerate(zip(r, optimal)):
        idx_rew = jnp.where(((r_i.cumsum()) % trial_length == 0) * r_i)[0]
        if len(idx_rew) == 0:
            idx_rew = jnp.array([len(r_i) - 1])

        idx_off = jnp.array([0] + idx_rew[:-1].tolist())
        dists = idx_rew - idx_off

        dists = dists / o_i

        all_steps.append(dists.mean())

    return jnp.array(all_steps)


def get_relative_steps_per_subgoal(r, locs, n_subgoals=40, large_maze=True):
    all_steps_per_subgoal, all_locs_per_subgoal, optimal_steps = [], [], []

    for i, (l_i, r_i) in enumerate(zip(locs, r)):
        idcs_rew = np.where(((r_i.cumsum()) % 1 == 0) * r_i)[0]
        locs_rew = np.array(l_i)[idcs_rew]
        idcs_off = np.array([0] + idcs_rew[:-1].tolist())
        dists_per_subgoal = idcs_rew - idcs_off

        # Clip to n_subgoals (for easy processing, or set to a max value if it did not)
        # reach the subgoal in time
        dps_arr = np.ones(n_subgoals) * 3000
        dps_arr[: len(dists_per_subgoal)] = np.array(dists_per_subgoal)[:n_subgoals]

        all_locs_per_subgoal = np.ones((n_subgoals, 2)) * -1
        all_locs_per_subgoal[: len(dists_per_subgoal)] = np.array(locs_rew)[:n_subgoals]

        all_steps_per_subgoal.append(dps_arr)

        # Compute the optimal distance using nx
        env = maze.MazeEnvironment(3, 3, 1, True, jr.PRNGKey(i), large_maze=large_maze)
        graph = layout_to_nx(env.layout)
        l0 = tuple(l_i[0].tolist())
        optimal_dists_i = np.ones(n_subgoals) * 3000
        for j, l1 in enumerate(locs_rew[:n_subgoals]):
            p = len(nx.shortest_path(graph, source=l0, target=tuple(l1.tolist()))) - 1
            l0 = tuple(l1.tolist())
            optimal_dists_i[j] = p
        optimal_steps.append(optimal_dists_i)

    all_steps_per_subgoal = jnp.array(all_steps_per_subgoal)
    all_locs_per_subgoal = jnp.array(all_locs_per_subgoal)
    optimal_steps = jnp.array(optimal_steps)

    return all_steps_per_subgoal / optimal_steps


def get_mean_and_stderr(x):
    return x.mean(0), x.std(0) / jnp.sqrt(x.shape[0])


def create_average_reward_plot(
    filename,
    curves,
    title,
    optimal=None,
    normalised=False,
    show=False,
    n_steps=10_000,
    legloc=None,
    small_ratio=False,
    skip=50,
):
    fig, ax = create_plot("square", figsize=(3, 1.5), set_box_aspect=False)
    for c in curves:
        if normalised:
            cr = c["normalised_convolved_reward"]
        else:
            cr = c["convolved_reward"]

        if cr.shape[0] == 0:
            print(f"Did not find curves for {c['tag']}")
            continue

        mu, stds = get_mean_and_stderr(cr[:, :n_steps])
        x = jnp.arange(len(mu))
        mu, stds, x = mu[::skip], stds[::skip], x[::skip]
        ax.plot(
            x,
            mu,
            label=c["label"],
            color=c["color"],
            linewidth=0.75,
            linestyle=c.get("linestyle", "-"),
        )
        ax.fill_between(x, mu - stds, mu + stds, alpha=0.15, color=c["color"])

    ax.set_title(title)
    ax.set_xlabel("Time (step)", fontsize=8)
    if normalised:
        ax.set_ylabel("Normalized Reward", fontsize=8)
    else:
        ax.set_ylabel("Reward", fontsize=8)
    ax.grid("on", linestyle="--", alpha=0.25, linewidth=0.5)
    ax.legend(
        loc="lower left" if legloc is None else legloc,
        fontsize=5,
        edgecolor="white",
        ncol=3 if len(curves) > 4 else 1,
        handlelength=1,
    )

    if optimal is not None:
        ax.axhline(optimal, linestyle="--", linewidth=0.75, color="black")
        t = ax.text(
            0, 0.915 * optimal, "Optimal", fontsize=6, horizontalalignment="left"
        )
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="w", alpha=0.85)])

        # ylim = plt.gca().get_ylim()
        # plt.gca().set_ylim([ylim[0], 1.12 * optimal])

    if small_ratio:
        plt.subplots_adjust(bottom=0.25, top=0.85, left=0.20, right=0.95)
        plt.savefig(filename, dpi=300)
    else:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.85)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def write_ttest_table_square(filename, labels, values, value_name, task_name):
    def sci_notation_10x(num):
        exponent = int(f"{num:e}".split("e")[1])
        mantissa = num / (10**exponent)
        return f"${mantissa:.2f}" + " \cdot 10^{" + str(exponent) + "}$"

    if "abcd" in task_name.lower():
        env = "abcd"
    elif "abcb" in task_name.lower():
        env = "abcb"
    else:
        raise NotImplementedError(f"Does not recognise task: {task_name}")

    if "reward" in value_name.lower():
        figref = f"fig:rewardsboxplot_{env.lower()}"
    else:
        figref = f"fig:timesboxplot_{env.lower()}"
    figref = "{" + figref + "}"

    ttest = "\\texttt{scipy.stats.ttest\\_rel}"
    # the {value_name} between the tested models in the {task_name.upper()}
    caption_title = (
        "\\textbf{Significance test for " + value_name + " in the " + task_name + "}. "
    )
    caption = (
        caption_title
        + f"Results of a paired t-test ({ttest}). (*) indicates significance ($p < 0.05$), and (n.s.) denotes no significance. The corresponding box plot is depicted in Figure~\\ref{figref}."
    )

    f = open(filename, "w")

    f.write("\\begin{table}[h!]\n")
    f.write("\\centering \n")
    f.write("\\caption{" + caption + "}\n")
    f.write("\\begin{tabular}{" + "l" * (len(labels) + 1) + "}\n")
    f.write("\t\\toprule \n")

    toprow = []
    for la in labels:
        toprow.append("&")
        toprow.append("\\textbf{" + la + "}")
    toprow.append("\\\\\n")
    f.write("\t" + " ".join(toprow))
    f.write("\t\\midrule \n")

    for la, a in zip(labels, values.T):
        row = []
        row.append("\t\\textbf{" + la + "}")
        row.append("&")
        for lb, b in zip(labels, values.T):
            if la != lb:
                res = ttest_rel(a, b)
                p = res.pvalue
                v = "*" if p < 0.05 else "n.s."
                # row.append(f"{v} ({sci_notation_10x(p)}) &")
                row.append("\small{" + f"{sci_notation_10x(p)} ({v})" + "}")
                row.append("&")
            else:
                row.append("&")
        row = row[:-1]
        row.append("\\\\\n")
        f.write(" ".join(row))

    f.write("\t\\bottomrule \n")
    f.write("\end{tabular}\n")
    f.write("\label" + figref.replace("fig", "tab") + "\n")
    f.write("\end{table}")

    f.close()


def write_ttest_table(filename, labels, values, value_name, task_name):
    def sci_notation_10x(num):
        exponent = int(f"{num:e}".split("e")[1])
        mantissa = num / (10**exponent)
        return f"${mantissa:.2f}" + " \cdot 10^{" + str(exponent) + "}$"

    if "abcd" in task_name.lower():
        env = "abcd"
    elif "abcb" in task_name.lower():
        env = "abcb"
    else:
        raise NotImplementedError(f"Does not recognise task: {task_name}")

    if "reward" in value_name.lower():
        figref = f"fig:rewardsboxplot_{env.lower()}"
    else:
        figref = f"fig:timesboxplot_{env.lower()}"
    figref = "{" + figref + "}"

    ttest = "\\texttt{scipy.stats.ttest\\_rel}"
    caption = f"Results of a paired t-test ({ttest}) comparing the {value_name} between the tested models in the {task_name.upper()}. The corresponding box plot is depicted in Figure~\\ref{figref}."

    f = open(filename, "w")

    f.write("\\begin{table}[h!]\n")
    f.write("\\centering \n")
    f.write("\\caption{" + caption + "}\n")
    f.write("\\begin{tabular}{lllc}\n")
    f.write("\t\\toprule \n")
    f.write(
        "\t\\textbf{Model 1} & \\textbf{Model 2} & \\textbf{P-value} & \\textbf{Significance} \\\\\n"
    )
    f.write("\t\\midrule \n")

    row_template = "\t{} & {} & {} & {} \\\\\n"
    for la, a in zip(labels, values.T):
        for lb, b in zip(labels, values.T):
            if la != lb:
                stat, p = ttest_rel(a, b)
                f.write(
                    row_template.format(
                        la, lb, sci_notation_10x(p), "*" if p < 0.05 else "n.s."
                    )
                )
        if la != labels[-1]:
            f.write("\t\\midrule \n")

    f.write("\t\\bottomrule \n")
    f.write("\end{tabular}\n")
    f.write("\label" + figref.replace("fig", "tab") + "\n")
    f.write("\end{table}")

    f.close()


def create_box_plot(
    filename, curves, key, ylabel, title, show=False, table=False, lastn=None
):
    values = np.array([c[key] for c in curves]).T
    labels = np.array([c["label"] for c in curves])

    if lastn is not None:
        values = values[-lastn:]

    if table:
        write_ttest_table_square(
            str(filename).replace(".pdf", ".tex"), labels, values, ylabel, title
        )

    fig, ax = create_plot("square", set_box_aspect=False, figsize=(2, 2))
    ax.grid("on", linestyle="--", alpha=0.25, linewidth=0.5)
    ax = sns.boxplot(
        values,
        showfliers=False,
        palette=["white"] * values.shape[1],
        boxprops=dict(edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.0),
        flierprops=dict(
            marker="o", markerfacecolor="gray", markeredgecolor="black", markersize=5
        ),
    )

    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_title(title)

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=8)

    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def get_grounding_prior(experiment_path):
    files = list(experiment_path.glob("*"))
    files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))
    all_gps = []
    for f in files:
        gps = dict(np.load(f))["grounding_prior"]
        all_gps.append(gps[None])
    return jnp.concatenate(all_gps, axis=0)


def extract_grounding_mat(mat):
    obs = jnp.where(mat.mean(0) > 0.5)[0]
    states = jnp.where(mat.mean(1) > 0.5)[0]
    # Normalise the matrix
    mat = mat / mat.sum(axis=-1, keepdims=True)
    mat = mat[states][:, obs]
    return mat, obs, states


def create_remapping_plots(filename, grounding_priors, alternation=False):
    mat0, obs0, states0 = extract_grounding_mat(grounding_priors[0, 0])
    mat1, obs1, states1 = extract_grounding_mat(grounding_priors[1, 0])

    fig, ax = plt.subplots(1, 2, figsize=(2, 2), dpi=300)

    # ax[0].set_aspect(2)
    # ax[1].set_aspect(2)

    ax[0].set_title("Block 1")
    ax[1].set_title("Block 2")
    ax[0].imshow(mat0, cmap="gray", aspect="auto")
    ax[1].imshow(mat1, cmap="gray", aspect="auto")
    ax[0].set_xticks(np.arange(len(obs0)), obs0, rotation=0, fontsize=7.5)
    ax[1].set_xticks(np.arange(len(obs1)), obs1, rotation=0, fontsize=7.5)

    if len(states0) != 4:
        ax[0].set_yticks(np.arange(len(states0)), np.arange(len(states0)), fontsize=7.5)
    else:
        labs = ["A", "B", "C", "D"]
        if alternation:
            labs[-1] = "B'"
        ax[0].set_yticks(np.arange(len(states0)), labs, fontsize=7.5)

    ax[1].set_yticks([])
    [a.set_xlabel("Location", fontsize=8) for a in ax.flatten()]
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(filename, dpi=300)
    plt.close()


def create_remapping_plots_alternation(filename, grounding_priors, mat_as):
    fig, ax = plt.subplots(1, 2, figsize=(2, 2), dpi=300)
    ax[0].set_title("Block 1")
    ax[1].set_title("Block 2")

    # ax[0].set_aspect(3)
    # ax[1].set_aspect(3)

    for i in range(2):
        a, g = mat_as[i][0], grounding_priors[i][0]
        f = jnp.arange(a.shape[0]) % 2 == 1
        mat = g.T @ a[f]
        mat, s, o = extract_grounding_mat(mat)
        idcs = np.array([1, 2, 0, 3])  # Manually set for visualisation purposes
        ax[i].imshow(mat.T[idcs], cmap="gray", aspect="auto")
        ax[i].set_xticks(np.arange(len(o)), o.tolist(), fontsize=7.5)

    [a.set_yticks(np.arange(4), ["A ", "B ", "C ", "B'"], fontsize=7.5) for a in ax]
    ax[1].set_yticks([])
    ax[0].set_xlabel("Location", fontsize=8)
    ax[1].set_xlabel("Location", fontsize=8)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(filename, dpi=300)
    plt.close()


def create_relative_distance_plot(filename, relative_distances, title):
    fig, ax = create_plot("square", set_box_aspect=False, figsize=(2, 2))
    ax.grid("on", linestyle="--", alpha=0.25, linewidth=0.5)

    n = relative_distances.shape[1]
    ax.axhline(6.44, color="black", linestyle="--", linewidth=0.75)
    t = ax.text(n - 0.5, 6.95, "Chance", fontsize=6, horizontalalignment="right")
    t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="w", alpha=0.85)])

    ax.axhline(1, color="black", linestyle="--", linewidth=0.75)
    t = ax.text(n - 0.5, -0.25, "Optimal", fontsize=6, horizontalalignment="right")
    t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="w", alpha=0.85)])

    mu, stds, color = *get_mean_and_stderr(relative_distances[:5]), "teal"
    ax.plot(mu, label="Blocks 1-5", linewidth=0.75, color=color)
    ax.fill_between(np.arange(len(mu)), mu - stds, mu + stds, alpha=0.05, color=color)

    mu, stds, color = *get_mean_and_stderr(relative_distances[-5:]), "red"
    ax.plot(mu, label="Blocks 35-40", linewidth=0.75, color=color)
    ax.fill_between(np.arange(len(mu)), mu - stds, mu + stds, alpha=0.05, color=color)

    ax.set_xlabel("Trial", fontsize=8)
    ax.set_ylabel("Relative distance", fontsize=8)

    ylim = ax.get_ylim()
    ax.set_ylim([-1.25, ylim[1]])

    ax.set_title(title)

    ax.legend(
        loc="upper right",
        ncol=1,
        fancybox=True,
        edgecolor="white",
        framealpha=1,
        fontsize=6,
        handlelength=1,
    )

    plt.savefig(filename, dpi=300)
    plt.close()


def get_curves(task, sim_path, repeat_envs=False, large_maze=True):
    if task == "abcd":
        curves = [
            {
                "label": "S-HAI K",
                "color": "chocolate",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
            },
            {
                "label": "S-HAI L",
                "color": "tab:orange",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_1_clone",
                "linestyle": "--",
            },
            {
                "label": "HAI-40",
                "color": "lightseagreen",
                "tag": "mixture:false_randomhigh:false_remap:false_learnschema:false_taskmodel:schema_40_80_clone",
            },
            {
                "label": "HAI-20",
                "color": "lightblue",
                "tag": "mixture:false_randomhigh:false_remap:false_learnschema:false_taskmodel:schema_20_40_clone",
            },
            # {
            #     "label": "HAI-2",
            #     "color": "tab:blue",
            #     "tag": "mixture:false_randomhigh:false_remap:false_learnschema:false_taskmodel:schema_2_4_clone",
            # },
            {
                "label": "Random",
                "color": "gray",
                "tag": "mixture:false_randomhigh:true_remap:false_learnschema:false_taskmodel:schema_1_1_clone",
            },
        ]
    elif task == "abcb":
        curves = [
            {
                "label": "S-HAI-2C K",
                "color": "purple",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_2_clone",
            },
            {
                "label": "S-HAI-2C L",
                "color": "magenta",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_2_clone",
                "linestyle": "--",
            },
            {
                "label": "S-HAI K",
                "color": "chocolate",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
            },
            {
                "label": "S-HAI L",
                "color": "tab:orange",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_1_clone",
                "linestyle": "--",
            },
            {
                "label": "Random",
                "color": "gray",
                "tag": "mixture:false_randomhigh:true_remap:false_learnschema:false_taskmodel:schema_1_1_clone",
            },
            # {
            #     "label": "S-HAI-2C L",
            #     "color": "purple",
            #     "tag": "mixture:false_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_2_clone",
            #     "linestyle": "--",
            # },
        ]
    elif task == "abcd_sim3":
        curves = [
            {
                "label": "S-HAI K",
                "color": "chocolate",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
            },
            {
                "label": "S-HAI L",
                "color": "tab:orange",
                "tag": "mixture:false_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_1_clone",
                "linestyle": "--",
            },
            {
                "label": "S-HAI K (MoGL)",
                "color": "tab:red",
                "tag": "mixture:true_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
            },
            {
                "label": "S-HAI L (MoGL)",
                "color": "tab:red",
                "tag": "mixture:true_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_1_clone",
                "linestyle": "--",
            },
        ]

    else:
        raise ValueError(f"Did not find curves for task: {task}")

    optimal_steps_per_trial, optimal_reward_rate = get_optimal_reward_rate_abcd(
        alternation=task == "abcb", repeat_envs=repeat_envs, large_maze=large_maze
    )

    # Compute some statistics that are useful for plotting
    print("Loading data from simulations ... (this might take some time)")
    for c in tqdm(curves):
        r, cr = get_rewards(sim_path / c["tag"])
        if len(r.shape) == 1:
            print(sim_path / c["tag"], " not found")
            del c
            continue

        locs = get_locations(sim_path / c["tag"])
        c["agent_location"] = locs
        c["convolved_reward"] = cr
        c["normalised_convolved_reward"] = cr / (4 / optimal_steps_per_trial[:, None])
        c["reward"] = r
        c["cumulative_reward"] = r.sum(-1)
        c["steps_to_first"] = get_num_steps_to_first(r)
        c["steps_per_trial"] = get_steps_per_trial(r)
        c["rel_steps_per_trial"] = get_steps_per_trial(r, optimal_steps_per_trial)
        c["rel_steps_per_subgoal"] = get_relative_steps_per_subgoal(
            r, locs, n_subgoals=20 * 4, large_maze=large_maze
        )
        c["schema_A"] = get_schema_likelihoods(sim_path / c["tag"])

    return curves, optimal_reward_rate


if __name__ == "__main__":
    root_path = Path(abcd.__file__).parent.parent

    ap = ArgumentParser()
    ap.add_argument("--load_from", type=str, default="data/maze-simulations")
    ap.add_argument("--store_at", type=str, default="data/paper_figures")
    ap.add_argument("--large_maze", action="store_true")
    ap.add_argument("--alternation", action="store_true")
    ap.add_argument("--repeat_envs", action="store_true")
    args = ap.parse_args()

    args.task = "abcb" if args.alternation else "abcd"

    prefix = "large" if args.large_maze else "small"
    sufsuffix = "_repeats" if args.repeat_envs else ""
    env_name = f"{prefix}_maze_{args.task}" + sufsuffix
    sim_path = root_path / args.load_from / env_name

    rep = "_repeat" if args.repeat_envs else ""

    store_path = root_path / args.store_at / env_name
    store_path.mkdir(exist_ok=True, parents=True)

    curves, optimal_reward_rate = get_curves(
        args.task, sim_path, args.repeat_envs, large_maze=args.large_maze
    )

    # Create the relative distance plot for the S-HAI K agent
    c = [c for c in curves if c["label"] == "S-HAI K"][0]
    create_relative_distance_plot(
        store_path / f"relative_distance_shaik_{args.task}.pdf",
        c["rel_steps_per_subgoal"].reshape(-1, 20, 4).mean(-1),
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
    )

    # Create the relative distance plot for the S-HAI L agent
    c = [c for c in curves if c["label"] == "S-HAI L"][0]
    create_relative_distance_plot(
        store_path / f"relative_distance_shail_{args.task}.pdf",
        c["rel_steps_per_subgoal"].reshape(-1, 20, 4).mean(-1),
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
    )

    # Create the relative distance plot for the S-HAI L agent
    c = [c for c in curves if c["label"] == "S-HAI-2C L"]
    if len(c) > 0:
        c = c[0]
        create_relative_distance_plot(
            store_path / f"relative_distance_shai2cl_{args.task}.pdf",
            c["rel_steps_per_subgoal"].reshape(-1, 20, 4).mean(-1),
            f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        )

    # Create the relative distance plot for the S-HAI L agent
    c = [c for c in curves if c["label"] == "S-HAI-2C K"]
    if len(c) > 0:
        c = c[0]
        create_relative_distance_plot(
            store_path / f"relative_distance_shai2ck_{args.task}.pdf",
            c["rel_steps_per_subgoal"].reshape(-1, 20, 4).mean(-1),
            f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        )

    create_average_reward_plot(
        store_path / f"average_rewards_{args.task}.pdf",
        curves,
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        optimal=optimal_reward_rate,
    )

    create_average_reward_plot(
        store_path / f"average_normalised_rewards_{args.task}.pdf",
        curves,
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        optimal=None,
        normalised=True,
        n_steps=10_000,
        # legloc="lower right",
    )

    # create_box_plot(
    #     store_path / f"steps_to_first_box_{args.task}.pdf",
    #     curves[::-1],
    #     "steps_to_first",
    #     "Steps to First Reward",
    #     f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
    # )

    create_box_plot(
        store_path / f"steps_per_trial_box_{args.task}.pdf",
        curves[::-1],
        "steps_per_trial",
        "Steps per Trial",
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        table=True,
    )

    create_box_plot(
        store_path / f"relative_steps_per_trial_box_{args.task}.pdf",
        curves[::-1],
        "rel_steps_per_trial",
        "Relative Steps per Trial",
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        table=True,
    )

    create_box_plot(
        store_path / f"cumulative_reward_box_{args.task}.pdf",
        curves[::-1],
        "cumulative_reward",
        "Cumulative Reward",
        f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        table=True,
    )

    if args.alternation:
        # Hardcoded path for S-HAI-2C K
        c = [c for c in curves if c["label"] == "S-HAI-2C K"][0]
        print((sim_path / c["tag"]).exists())
        grounding_priors = get_grounding_prior(sim_path / c["tag"])
        create_remapping_plots_alternation(
            store_path / f"grounding_prior_{args.task}_1_2.pdf",
            grounding_priors,
            np.array(c["schema_A"]),
        )
    else:
        # Hardcoded path for S-HAI L
        c = [c for c in curves if c["label"] == "S-HAI L"][0]
        print((sim_path / c["tag"]).exists())
        grounding_priors = get_grounding_prior(sim_path / c["tag"])
        create_remapping_plots(
            store_path / f"grounding_prior_{args.task}_1_1.pdf", grounding_priors
        )

    if args.task == "abcd":
        # This is for simulation 3, where we compare MoG vs normal
        curves_sim3, _ = get_curves(
            "abcd_sim3", sim_path, args.repeat_envs, large_maze=args.large_maze
        )
        rep = "_repeat" if args.repeat_envs else ""
        create_average_reward_plot(
            store_path / f"average_normalised_rewards_abcd{rep}_mog.pdf",
            curves_sim3,
            f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
            optimal=None,
            normalised=True,
            n_steps=3000,
            legloc="lower right",
            small_ratio=True,
        )

        c = [c for c in curves_sim3 if c["label"] == "S-HAI L (MoGL)"][0]
        create_relative_distance_plot(
            store_path / f"relative_distance_shail_{args.task}_mog.pdf",
            c["rel_steps_per_subgoal"],
            f"{args.task.upper()} Task" + (" (Repeat)" if args.repeat_envs else ""),
        )
