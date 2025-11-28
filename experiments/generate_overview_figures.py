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

import matplotlib.pyplot as plt

from generate_neural_figures import get_env_info, plot_env

from pathlib import Path

import abcd

from matplotlib.colors import ListedColormap

from abcd.visualize import get_graph
import igraph

import pickle
import numpy as np

from train_cscgs_offline import generate_data

if __name__ == "__main__":
    root_path = Path(abcd.__file__).parent.parent

    store_path = root_path / "data/paper_figures"
    store_path.mkdir(exist_ok=True, parents=True)

    for large in [True, False]:
        prefix = "large" if large else "small"

        fig, ax = plt.subplots(1, 2, figsize=(2, 1.15), dpi=300)
        for i in range(2):
            layout, reward_options, reward_labels = get_env_info(i, large)
            plot_env(ax[i], layout, reward_options, reward_labels, alpha=0.75)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"ABCD Block {i + 1}", fontsize=6)
        plt.savefig(
            store_path / f"{prefix}_abcd_task.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(2, 1.15), dpi=300)
        for i in range(2):
            layout, reward_options, reward_labels = get_env_info(i, large)
            plot_env(ax[i], layout, reward_options[:-1], reward_labels[:-1], alpha=0.75)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"ABCB Block {i + 1}", fontsize=6)
        plt.savefig(
            store_path / f"{prefix}_abcb_task.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()

    colors = ["#ffffff", "#edae49", "#d1495b", "#00798c", "#003d5b", "#218655"]
    env_cmap = ListedColormap(colors, name="my_cmap")

    load_path = root_path / "data/learned-models-offline"
    with open(load_path / "large_maze_navigation.pickle", "rb") as f:
        nav_chmm = pickle.load(f)

    o, r, a, s = generate_data(
        10000,
        switch_env_every=10000,
        n_envs_to_learn=1,
        alternation=False,
        large_maze=True,
    )

    g, v = get_graph(
        nav_chmm, np.array(o).astype(np.int64), np.array(a).astype(np.int64), False
    )

    print(np.unique(o))

    node_labels = np.arange(5).repeat(100)[v]

    cmap = plt.get_cmap("viridis")
    colors = [
        env_cmap(nl, alpha=0.85) for nl in (1 + node_labels) / (node_labels.max() + 1)
    ]

    igraph.plot(
        g,
        store_path / "low_level_graph.pdf",
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_size=28,
        margin=40,
        edge_arrow_size=0.5,
    )

    igraph.plot(
        g,
        store_path / "low_level_graph_labeled.pdf",
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=28,
        margin=40,
        edge_arrow_size=0.5,
    )
