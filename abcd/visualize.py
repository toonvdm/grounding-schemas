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

import igraph
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
import jax.numpy as jnp


def get_graph(chmm, x, a, multiple_episodes=False):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)[::-1]
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A.astype(np.float16) > 1e-4).tolist())
    return g, v


def plot_graph(
    chmm,
    x,
    a,
    output_file,
    n_uncloned_states,
    cmap=cm.Spectral,
    multiple_episodes=False,
    vertex_size=30,
    n_clones=1,
):
    g, v = get_graph(chmm, x, a, multiple_episodes)

    node_labels = np.arange(n_uncloned_states).repeat(n_clones)[v]

    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in (1 + node_labels) / (node_labels.max() + 1)]

    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        # layout=g.layout("grid"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out


def add_significance_bars(ax, comparisons, box_data, line_height=0.2, spacing=0.05):
    # Get max Y value from the top whiskers to start
    y_base = max([line.get_ydata()[1] for line in box_data["whiskers"]])
    stack_level = 0
    occupied_levels = []

    for comp in sorted(comparisons, key=lambda x: (x[0], x[1])):
        x1, x2, p = comp
        y = y_base + line_height * (stack_level + 1)

        # Determine stars
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "ns"

        # Avoid overlap by checking existing levels
        while any(abs(y - lvl) < spacing for lvl in occupied_levels):
            stack_level += 1
            y = y_base + line_height * (stack_level + 1)

        occupied_levels.append(y)

        # Draw the line and text
        ax.plot([x1, x1, x2, x2], [y, y + spacing, y + spacing, y], lw=1.5, c="k")
        ax.text((x1 + x2) * 0.5, y + spacing, stars, ha="center", va="bottom")


def add_horizontal_significance_bars(
    ax, comparisons, box_data, line_height=0.2, spacing=0.05
):
    # Get max x values from the right whiskers
    x_base = max([line.get_xdata()[1] for line in box_data["whiskers"]]) + 5
    stack_level = 0
    occupied_levels = []

    for comp in sorted(comparisons, key=lambda x: (x[0], x[1])):
        y1, y2, p = comp
        y_min, y_max = sorted([y1, y2])
        x = x_base + line_height * (stack_level + 1)

        # Determine significance stars
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "ns"
            continue

        # Avoid overlap
        while any(abs(x - lvl) < spacing for lvl in occupied_levels):
            stack_level += 1
            x = x_base + (stack_level + 1)

        occupied_levels.append(x)

        # Draw line and text
        ax.plot([x, x], [y_min - 0.1, y_max - 0.1], lw=0.5, c="k")
        ax.text(x + 0.1, (y1 + y2) / 2, stars, va="center", ha="left", rotation=90)


def plot_w_err(ax, rh, color, label, plot_error=False, **kwargs):
    means = jnp.array([np.mean(ri) for ri in rh])
    stds = jnp.array([np.std(ri) / np.sqrt(len(ri)) for ri in rh]) + 1e-8
    ax.plot(means, label=label, color=color, **kwargs)
    if plot_error:
        ax.fill_between(
            np.arange(len(means)),
            means - stds,
            means + stds,
            alpha=0.06,
            color=color,
        )
