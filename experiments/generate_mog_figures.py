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

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from argparse import ArgumentParser

import abcd

from generate_simulation_figures import (
    extract_grounding_mat,
    get_grounding_prior,
    create_plot,
)


def create_qz_ell_plot(filename, qzs, ells, show=False):
    qzs = qzs[:, :, :45]

    block_length = qzs.shape[1]
    n_blocks = qzs.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(4, 4), dpi=300, sharex=True)

    ax = axes[0]

    ax.imshow(
        qzs.reshape(-1, qzs.shape[-1]).T,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_ylabel(r"$q(z)$", fontsize=8)
    ax.set_xticks(
        np.arange(0, n_blocks + 1, 5) * block_length,
        np.arange(0, n_blocks + 1, 5),
        fontsize=8,
    )
    ax.set_yticks(np.arange(0, 49, 10), np.arange(0, 49, 10), fontsize=8)

    for i in range(40):
        axes[0].axvline(
            i * block_length, color="tab:blue", linestyle="--", linewidth=0.5
        )
        axes[1].axvline(i * block_length, color="black", linestyle="--", linewidth=0.5)

    ax = axes[1]
    ax.set_ylabel(r"$\text{E}[\log \text{P}]$", fontsize=8)
    ax.plot(ells, linewidth=1.0)
    ax.set_xticks(
        np.arange(0, n_blocks + 1, 5) * block_length,
        np.arange(0, n_blocks + 1, 5),
        fontsize=8,
    )

    treshold = -5
    t = ax.text(
        len(ells) - 5000,
        treshold - 2,
        "Threshold",
        fontsize=6,
        horizontalalignment="right",
        color="crimson",
    )
    ax.axhline(treshold, linestyle="--", color="crimson")
    ax.set_ylim([-10, ax.get_ylim()[1]])

    # We change the fontsize of minor ticks label
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="w", alpha=0.85)])

    ax.set_xlabel("Block")

    fig.align_ylabels(axes)

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def create_ell_plot(filename, ells, show=False):
    fig, ax = create_plot("square", figsize=(3, 1.5), set_box_aspect=False)
    ax.set_ylabel(r"$\text{E}[\log \text{P}]$", fontsize=8)
    ax.plot(ells, linewidth=0.75)
    n_blocks = 40
    block_length = 10000

    ax.set_xticks(
        np.arange(0, n_blocks + 1, 5) * block_length,
        np.arange(0, n_blocks + 1, 5),
        fontsize=8,
    )

    treshold = -5
    t = ax.text(
        len(ells) - 5000,
        treshold - 1,
        "Threshold",
        fontsize=6,
        horizontalalignment="right",
        color="crimson",
    )
    ax.axhline(treshold, linestyle="--", color="crimson")
    ax.set_ylim([-10, ax.get_ylim()[1]])

    # We change the fontsize of minor ticks label
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="w", alpha=0.85)])

    for i in range(40):
        ax.axvline(
            i * block_length, color="black", linestyle="--", linewidth=0.5, alpha=0.5
        )

    ax.set_xlabel("Block", fontsize=8)

    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.20, right=0.95)
    plt.savefig(filename, dpi=300)

    if show:
        plt.show()
    plt.close()


def create_qz_plot(filename, qzs, show=False):
    qzs = qzs[:, :, :45]

    block_length = qzs.shape[1]
    n_blocks = qzs.shape[0]

    fig, ax = create_plot("square", figsize=(3, 1.5), set_box_aspect=False)

    ax.imshow(
        qzs.reshape(-1, qzs.shape[-1]).T,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_ylabel(r"$q(z)$", fontsize=8)
    ax.set_xticks(
        np.arange(0, n_blocks + 1, 5) * block_length,
        np.arange(0, n_blocks + 1, 5),
        fontsize=6,
    )
    ax.set_yticks(np.arange(0, 49, 10), np.arange(0, 49, 10), fontsize=6)

    for i in range(40):
        ax.axvline(i * block_length, color="tab:blue", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Block", fontsize=8)

    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.20, right=0.95)
    plt.savefig(filename, dpi=300)

    if show:
        plt.show()
    plt.close()


def create_mog_grounding_plot(filename, grounding_priors):
    fig = plt.figure(figsize=(10, 2), dpi=300)  # Adjust width as needed
    gs = gridspec.GridSpec(1, 6 * 4 + 1, figure=fig)
    axes = []
    for i in range(6):
        b = i * 4
        if i > 0:
            b += 1
        axes.append(fig.add_subplot(gs[0, b : b + 4]))

    # After 1 block
    m0, o0, s0 = extract_grounding_mat(grounding_priors[1, 0])
    if o0.shape[0] == 4:
        axes[0].imshow(m0, cmap="gray")
        axes[0].set_xticks(np.arange(len(o0)), o0)
        axes[0].set_yticks(np.arange(len(o0)), ["A", "B", "C", "D"])
        axes[0].set_xlabel("Location")

    # After 5 block
    for i in range(5):
        m0, o0, s0 = extract_grounding_mat(grounding_priors[5, i])
        if o0.shape[0] == 4:
            axes[i + 1].imshow(m0, cmap="gray")
            axes[i + 1].set_xticks(np.arange(len(o0)), o0)
            axes[i + 1].set_yticks([])
            axes[i + 1].set_xlabel("Location")

    axes[1].set_yticks(np.arange(4), ["A", "B", "C", "D"])

    axes[0].set_title("After 1 Block", fontsize=13)
    axes[3].set_title("After 5 Blocks", fontsize=13)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def load_qz_ell(sim_path):
    ells, qzs = [], []
    for i in range(40):
        d = dict(np.load(sim_path / f"episode_{i}.npz"))
        ells.append(d["max_ell"])
        qzs.append(d["qz"])
    qzs = np.array(qzs)
    ells = np.concatenate(ells)
    return qzs, ells


if __name__ == "__main__":
    root_path = Path(abcd.__file__).parent.parent

    ap = ArgumentParser()
    ap.add_argument("--load_from", type=str, default="data/maze-simulations")
    ap.add_argument("--store_at", type=str, default="data/paper_figures")
    ap.add_argument("--alternation", action="store_true")
    ap.add_argument("--repeat_envs", action="store_true")
    ap.add_argument("--large_maze", action="store_true")
    args = ap.parse_args()

    args.task = "abcb" if args.alternation else "abcd"

    prefix = "large" if args.large_maze else "small"
    sufsuffix = "_repeats" if args.repeat_envs else ""
    env_name = f"{prefix}_maze_{args.task}" + sufsuffix
    sim_path = root_path / args.load_from / env_name
    store_path = root_path / args.store_at / env_name / "mogl"
    store_path.mkdir(exist_ok=True, parents=True)

    runs = [
        {
            "label": "S-HAI K (MoG)",
            "color": "tab:red",
            "tag": "mixture:true_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
        },
        {
            "label": "S-HAI L (MoG)",
            "color": "tab:red",
            "tag": "mixture:true_randomhigh:false_remap:true_learnschema:true_taskmodel:schema_1_1_clone",
            "linestyle": "--",
        },
    ]

    for r in runs:
        name = (
            r["label"]
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )
        qzs, ells = load_qz_ell(sim_path / r["tag"])
        create_qz_ell_plot(store_path / (name + "_qzell.pdf"), qzs, ells)

        create_qz_plot(store_path / (name + "_qz.pdf"), qzs)
        create_ell_plot(store_path / (name + "_ell.pdf"), ells)

        gp = get_grounding_prior(sim_path / r["tag"])
        create_mog_grounding_plot(store_path / (name + "_grounding_mog.pdf"), gp)
