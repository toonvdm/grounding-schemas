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

learn_clone_graphs: 
	python experiments/train_cscgs_offline.py --large_maze --n_nav_clones=100
	python experiments/train_cscgs_offline.py --large_maze --n_nav_clones=100 --load_navigation --alternation
	python experiments/train_cscgs_offline.py --n_nav_clones=1 
	python experiments/train_cscgs_offline.py --n_nav_clones=1 --load_navigation --alternation

large_maze_experiments: 
	python experiments/run_maze_experiment.py --large_maze
	python experiments/run_maze_experiment.py --large_maze --alternation
	python experiments/run_maze_experiment.py --large_maze --repeat_envs

small_maze_experiments:
	python experiments/run_maze_experiment.py 
	python experiments/run_maze_experiment.py --alternation
	python experiments/run_maze_experiment.py --repeat_envs

large_maze_paper_figures: 
	python experiments/generate_simulation_figures.py --large_maze
	python experiments/generate_simulation_figures.py --large_maze --repeat_envs
	python experiments/generate_simulation_figures.py --large_maze --alternation
	python experiments/generate_mog_figures.py --large_maze
	python experiments/generate_mog_figures.py --large_maze --repeat_envs
	python experiments/generate_neural_figures.py --large_maze
	python experiments/generate_simulation_gif.py --large_maze
	python experiments/generate_overview_figures.py

small_maze_paper_figures: 
	python experiments/generate_simulation_figures.py 
	python experiments/generate_simulation_figures.py --repeat_envs
	python experiments/generate_simulation_figures.py --alternation
	python experiments/generate_mog_figures.py
	python experiments/generate_mog_figures.py --repeat_envs
	python experiments/generate_neural_figures.py
	python experiments/generate_simulation_gif.py

experiments: large_maze_experiments small_maze_experiments
figures: large_maze_paper_figures small_maze_paper_figures 
all: learn_clone_graphs experiments figures

debug: 
	python experiments/run_maze_experiment.py --debug --verbose --large_maze