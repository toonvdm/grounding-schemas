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

from pathlib import Path
from argparse import ArgumentParser
import numpy as np

from tqdm import trange

import jax.random as jr

import abcd
from abcd import maze

import matplotlib.pyplot as plt

import mediapy
import io

from generate_simulation_figures import get_locations, get_rewards
from generate_neural_figures import plot_env


def fig2img(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, facecolor="white", format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im[:, :, :3]


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


if __name__ == "__main__":
    root_path = Path(abcd.__file__).parent.parent

    parser = ArgumentParser()
    parser.add_argument("--load_from", type=str, default="data/maze-simulations")
    parser.add_argument("--store_at", type=str, default="data/visuals")
    parser.add_argument("--alternation", action="store_true")
    parser.add_argument("--large_maze", action="store_true")
    args = parser.parse_args()

    prefix = "large" if args.large_maze else "small"
    suffix = "abcb" if args.alternation else "abcd"
    env_key = f"{prefix}_maze_{suffix}"
    sim_path = root_path / args.load_from / env_key

    store_path = root_path / args.store_at
    store_path.mkdir(exist_ok=True, parents=True)

    curve = {
        "label": "S-HAI K",
        "color": "tab:orange",
        "tag": "mixture:false_randomhigh:false_remap:true_learnschema:false_taskmodel:schema_1_1_clone",
        "linestyle": "--",
    }

    # curve = {"label": "S-HAI-2C L", "tag": "../../debug"}

    locations = get_locations(sim_path / curve["tag"])
    print(len(locations))

    rewards, _ = get_rewards(sim_path / curve["tag"])
    print(rewards.sum())
    print((sim_path / curve["tag"]).absolute())

    figs = []
    env_infos = [get_env_info(i, args.large_maze) for i in range(5)]
    print([rewards[i].sum() for i in range(5)])
    for t_off in trange(250):
        t = t_off + 5000

        env_infos = [get_env_info(i, args.large_maze) for i in range(5)]
        fig, ax = plt.subplots(1, 5, figsize=(4.5, 1.0), dpi=300)
        [a.set_xticks([]) for a in ax.flatten()]
        [a.set_yticks([]) for a in ax.flatten()]

        for i in range(5):
            a = ax.flatten()[i]
            plot_env(a, *env_infos[i], alpha=0.45)
            a.scatter(
                locations[i][t][0],
                locations[i][t][1],
                marker=".",
                color="tab:red",
                s=15,
                zorder=10,
            )
            a.set_title(f"Block {i + 1}")

        plt.subplots_adjust(bottom=0.0, top=0.8, left=0.05, right=0.95)
        # plt.show()
        figs.append(fig2img(fig))
        plt.close()

    with mediapy.set_show_save_dir(store_path):
        mediapy.show_videos({f"mazes_{env_key}": figs}, fps=40, codec="gif", width=1800)
