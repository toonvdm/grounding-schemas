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
import equinox as eqx


import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array


def create_small_maze_layout(reward_maze=False):
    layout = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    if reward_maze:
        layout = jnp.ones_like(layout)
    return layout


def create_large_maze_layout(
    nrows=3, ncols=3, cs=2, outer_wall=False, reward_maze=False
):
    """
    cs: corridor_size
    """
    maze = np.ones((nrows * 3 + (nrows - 1) * cs, ncols * 3 + (ncols - 1) * cs)) * (-1)

    well_min = 0
    well_max = 2
    corridor_value = well_max + 1

    outer_wall_value = corridor_value + 1
    outer_wall_value_vert = corridor_value + 1

    key = jr.PRNGKey(0)
    for i in range(ncols):
        for j in range(nrows):
            key, subkey = jr.split(key)
            well = jr.randint(subkey, (3, 3), minval=well_min, maxval=well_max + 1)
            if reward_maze:
                well = well.at[1, 1].set(-2)
            r0 = (j * cs) + j * 3
            c0 = (i * cs) + i * 3
            maze[r0 : r0 + 3, c0 : c0 + 3] = well

    start = 1
    for r in range(nrows):
        i = start + r * (cs + 3)
        for j in range(ncols - 1):
            s0 = 3 + j * (3 + cs)
            maze[i, s0 : s0 + cs] = corridor_value

    for c in range(ncols):
        i = start + c * (cs + 3)
        for j in range(nrows - 1):
            s0 = 3 + j * (3 + cs)
            maze[s0 : s0 + cs, i] = corridor_value

    if outer_wall:
        new_maze = np.ones(np.array(maze.shape) + 2) * -1
        new_maze[1:-1, 1:-1] = maze

        for c in range(ncols):
            i = start + c * (cs + 3) + 1
            new_maze[0, i] = outer_wall_value_vert  # + c
            new_maze[-1, i] = outer_wall_value_vert  # + c

        for r in range(nrows):
            i = start + r * (cs + 3) + 1
            new_maze[i, 0] = outer_wall_value  # + r
            new_maze[i, -1] = outer_wall_value  # + r

        maze = new_maze

    new_maze = np.ones(np.array(maze.shape) + 2) * -1
    new_maze[1:-1, 1:-1] = maze
    maze = new_maze

    if reward_maze:
        maze = maze == -2

    return jnp.array(maze).astype(jnp.int32)


class MazeEnvironment(eqx.Module):
    agent_location: Array
    reward_idx: int
    reward_sequence: Array

    # These are technically static, but give warnings if I run them as static.
    # I should probably have a separate namedtuple for these, and pass that into
    # each method separately
    layout: Array
    reward_options: Array
    starting_options: Array

    alternation: bool = eqx.field(static=True)

    def __init__(
        self,
        nrows,
        ncols,
        corridor_size,
        outer_wall,
        key,
        large_maze=True,
        alternation=False,
    ):
        # Static setup
        if large_maze:
            layout = create_large_maze_layout(nrows, ncols, corridor_size, outer_wall)
            reward_layout = create_large_maze_layout(
                nrows, ncols, corridor_size, outer_wall, reward_maze=True
            )
        else:
            layout = create_small_maze_layout()
            reward_layout = create_small_maze_layout(reward_maze=True)

        self.layout = layout

        self.reward_options = jnp.concatenate(
            [i[:, None] for i in jnp.where(reward_layout == 1)], axis=-1
        )

        self.alternation = alternation

        # Specific Task configuration
        key, *subkeys = jr.split(key, 3)
        self.reward_idx = 0

        reward_sequence = jr.choice(
            subkeys[0],
            jnp.arange(self.reward_options.shape[0]),
            shape=(4,),
            replace=False,
        )
        if alternation:
            reward_sequence = reward_sequence.at[-1].set(reward_sequence[1])

        self.reward_sequence = reward_sequence

        self.starting_options = jnp.concatenate(
            [i[:, None] for i in jnp.where(self.layout != -1)], axis=-1
        )
        self.agent_location = jr.choice(subkeys[1], self.starting_options)

    def reset(self, key):
        key, *subkeys = jr.split(key, 3)
        reward_idx = 0
        reward_sequence = jr.choice(
            subkeys[0],
            jnp.arange(self.reward_options.shape[0]),
            shape=(4,),
            replace=False,
        )
        if self.alternation:
            reward_sequence = reward_sequence.at[-1].set(reward_sequence[1])

        agent_location = jr.choice(subkeys[1], self.starting_options)

        env = eqx.tree_at(lambda x: x.reward_idx, self, reward_idx)
        env = eqx.tree_at(lambda x: x.reward_sequence, env, reward_sequence)
        env = eqx.tree_at(lambda x: x.agent_location, env, agent_location)

        obs = self.layout[agent_location[0], agent_location[1]]

        return jnp.array([obs, 0]), env

    @staticmethod
    def action_map(action):
        # 0: right 1: Left 2: Down 4: Up
        return jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])[action]

    @jax.jit
    def act(self, action):
        agent_location = self.agent_location + self.action_map(action)

        # Make sure the agent does not go out of bounds
        agent_location = jnp.clip(
            agent_location,
            jnp.zeros(2),
            jnp.array([self.layout.shape[0] - 1, self.layout.shape[1] - 1]),
        ).astype(jnp.int32)

        # If the agent moves into a wall, it is invalid
        agent_location = jax.lax.cond(
            self.layout[agent_location[0], agent_location[1]] == -1,
            lambda: self.agent_location,
            lambda: agent_location,
        )

        obs = self.layout[agent_location[0], agent_location[1]]

        reward = jnp.all(
            agent_location == self.reward_options[self.reward_sequence[self.reward_idx]]
        )

        # If reward, we want to step the reward index
        reward_idx = jax.lax.cond(
            reward,
            lambda: (self.reward_idx + 1) % len(self.reward_sequence),
            lambda: self.reward_idx,
        )

        env = eqx.tree_at(lambda x: x.agent_location, self, agent_location)
        env = eqx.tree_at(lambda x: x.reward_idx, env, reward_idx)
        return jnp.array([obs, reward]), env
