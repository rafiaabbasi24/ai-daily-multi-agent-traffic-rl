# Dynamic Traffic Light Control with Multi‑Agent RL  

**Category:** Reinforcement Learning  
**Tech stack:** Python, Gymnasium, PyTorch, NumPy, Stable‑Baselines3  

---  

## Table of Contents  

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Training a Multi‑Agent Policy](#training-a-multi‑agent-policy)  
6. [Evaluation & Visualization](#evaluation--visualization)  
7. [Configuration](#configuration)  
8. [Project Structure](#project-structure)  
9. [Contributing](#contributing)  
10. [License](#license)  

---  

## Project Overview  

This repository implements a **multi‑agent reinforcement learning (MARL)** framework for adaptive traffic‑signal control.  
Each intersection is modelled as an independent agent that learns to minimize cumulative vehicle waiting time, queue length, and emissions by coordinating with its neighbours.  

The environment follows the **Gymnasium** API (`gymnasium.Env`) and can be swapped with custom traffic simulators (e.g., SUMO) without changing the learning pipeline.  

The learning pipeline is built on **Stable‑Baselines3** with a custom wrapper that enables parallel training of multiple agents using shared or independent policies.  

---  

## Features  

- **Modular environment** (`traffic_env/`) that conforms to the Gymnasium interface.  
- **Multi‑agent wrapper** (`multi_agent/`) that handles agent registration, observation/action splitting, and reward aggregation.  
- **Stable‑Baselines3 integration** with PPO, A2C, and custom MARL algorithms.  
- **Config‑driven experiments** (`configs/`) using JSON/YAML for reproducibility.  
- **Logging & checkpointing** via TensorBoard and Weights & Biases (optional).  
- **Evaluation scripts** that generate per‑intersection performance metrics and visualizations.  

---  

## Installation  

> **Note:** The project has been tested on Python 3.10 and Ubuntu 22.04.  

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-traffic-light-rl.git
cd dynamic-traffic-light-rl

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Optional: install extra packages for visualization
pip install -r requirements-visualization.txt
```

**`requirements.txt`** (core)

```
gymnasium==0.28.1
torch==2.2.0
numpy==1.26.2
stable-baselines3==2.2.1
pandas==2.2.1
PyYAML==6.0.1
tensorboard==2.16.2
```

**`requirements-visualization.txt`** (optional)

```
matplotlib==3.8.4
seaborn==0.13.2
networkx==3.2.1
```

---  

## Quick Start  

The easiest way to see the system in action is to run a pre‑trained policy on a small 4‑intersection grid.

```bash
# Download a sample checkpoint (provided in the repo)
wget -O checkpoints/sample_policy.zip https://example.com/sample_policy.zip
unzip checkpoints/sample_policy.zip -d checkpoints/

# Run the evaluation script
python scripts/evaluate.py \
    --config configs/eval_small_grid.yaml \
    --checkpoint checkpoints/sample_policy.zip
```

You should see a live Matplotlib window displaying the traffic flow and a summary printed to the console, e.g.:

```
Episode 0 | Avg. waiting time: 12.3 s | Throughput: 158 vehicles
```

---  

## Training a Multi‑Agent Policy  

### 1. Prepare a configuration file  

All hyper‑parameters are stored in `configs/`.  
Copy the template and adjust as needed:

```bash
cp configs/train_template.yaml configs/train_my_experiment.yaml
```

Key sections to edit:

- `env.grid_size`: number of intersections per dimension (e.g., `5` for a 5×5 grid).  
- `env.max_steps`: episode length in simulation steps.  
- `agents.type`: `"shared"` (single policy for all agents) or `"independent"` (separate policies).  
- `algorithm`: `"ppo"` (default) or `"a2c"`.  
- `training.total_timesteps`: total environment steps for training.  

### 2. Launch training  

```bash
python scripts/train.py \
    --config configs/train_my_experiment.yaml \
    --log_dir logs/my_experiment \
    --tensorboard
```

- **TensorBoard** logs are written to `logs/my_experiment/tb`.  
- Checkpoints are saved automatically every `save_interval` steps (default: 100 k).  

### 3. Monitor training  

```bash
tensorboard --logdir logs/my_experiment/tb
```

You will see curves for **episode reward**, **average waiting time**, and **policy entropy** for each agent (or the shared policy).  

---  

## Evaluation & Visualization  

### Command‑line evaluation  

```bash
python scripts/evaluate.py \
    --config configs/eval_large_grid.yaml \
    --checkpoint logs/my_experiment/checkpoints/last.zip \
    --render
```

- `--render` opens a live Matplotlib animation of vehicle movements.  
- The script writes a CSV file with per‑step metrics to `results/`.  

### Plotting utilities  

The repository includes helper functions in `utils/visualization.py`.  
Example usage:

```python
import pandas as pd
from utils.visualization import plot_waiting_time, plot_throughput

df = pd.read_csv('results/eval_large_grid_metrics.csv')
plot_waiting_time(df, save_path='figs/waiting_time.png')
plot_throughput(df, save_path='figs/throughput.png')
```

---  

## Configuration  

All experiments are driven by a single YAML file. Below is a minimal example (`configs/train_template.yaml`):

```yaml
env:
  name: traffic_env.MultiIntersectionEnv
  grid_size: 4               # 4x4 intersections
  max_steps: 3600            # 1 hour of simulated time (assuming 1 step = 1 sec)
  traffic_density: 0.25      # proportion of road cells occupied at start
  seed: 42

agents:
  type: shared               # "shared" or "independent"
  observation_space: "grid" # "grid" or "vector"
  action_space: "phase"      # "phase" (choose signal phase) or "duration"

algorithm:
  name: ppo
  policy: MlpPolicy
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01

training:
  total_timesteps: 5_000_000
  log_interval: 10
  eval_interval: 50_000
  save_interval: 100_000
  eval_episodes: 5
```

The same schema is used for evaluation (`configs/eval_*.yaml`) with additional fields such as `render: true`.  

---  

## Project Structure  

```
dynamic-traffic-light-rl/
├─ traffic_env/                # Gymnasium environment implementation
│   ├─ __init__.py
│   ├─ base.py                 # Core environment logic
│   └─ utils.py                # Helper functions (e.g., vehicle generation)
├─ multi_agent/                # Multi‑agent wrappers & utilities
│   ├─ __init__.py
│   ├─ wrapper.py              # Gymnasium wrapper that splits obs/action per agent
│   └─ policy.py               # Optional custom policy networks
├─ configs/                    # YAML configuration files
│   ├─ train_template.yaml
│   ├─ eval_small_grid.yaml
│   └─ eval_large_grid.yaml
├─ scripts/                    # Entry‑point scripts
│   ├─ train.py                # Training pipeline
│   └─ evaluate.py             # Evaluation pipeline
├─ utils/                      # Miscellaneous utilities
│   ├─ logger.py               # Unified logger (TensorBoard / WandB)
│   └─ visualization.py        # Plotting helpers
├─ logs/                       # Generated TensorBoard logs & checkpoints
├─ results/                    # CSV files from evaluation runs
├─ figures/                    # Exported plots (git‑ignored)
├─ requirements.txt
├─ requirements-visualization.txt
└─ README.md
```

---  

## Contributing  

Contributions are welcome! Please follow these steps:

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feat/your-feature`).  
3. Install the development dependencies:  

   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

4. Run the test suite (see `tests/` directory) to ensure everything passes:  

   ```bash
   pytest -vv
   ```

5. Submit a Pull Request with a clear description of the changes and any new configuration files.  

**Coding style:** Follow PEP 8, use type hints, and keep docstrings in NumPy style.  

---  

## License  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  