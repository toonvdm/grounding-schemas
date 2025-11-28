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


def lr2obs(loc, reward):
    # Map a location and reward to an flat observation
    return loc * 2 + reward


def obs2lr(obs):
    # Map a flat observation to a location and reward
    return obs // 2, obs % 2
