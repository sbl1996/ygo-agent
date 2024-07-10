# YGO Agent

YGO Agent is a project aimed at mastering the popular trading card game Yu-Gi-Oh! through deep learning. Based on a high-performance game environment (ygoenv), this project leverages reinforcement learning and large language models to develop advanced AI agents (ygoai) that aim to match or surpass human expert play. YGO Agent provides researchers and players with a platform for exploring AI in complex, strategic game environments.

[Discord](https://discord.gg/EqWYj4G4Ys)

## News🔥

- 2024.7.2 - We have a discord channel for discussion now! We are also working with [neos-ts](https://github.com/DarkNeos/neos-ts) to implement human-AI battle.
- 2024.4.18 - LSTM has been implemented and well tested.
- 2024.4.7 - We have switched to JAX for training and evaluation due to the better performance and flexibility.


## Table of Contents
- [Subprojects](#subprojects)
  - [ygoenv](#ygoenv)
  - [ygoai](#ygoai)
- [Installation](#installation)
  - [Building from source](#building-from-source)
  - [Troubleshooting](#troubleshooting)
- [Evaluation](#evaluation)
  - [Obtain a trained agent](#obtain-a-trained-agent)
  - [Play against the agent](#play-against-the-agent)
  - [Battle between two agents](#battle-between-two-agents)
- [Training](#training)
  - [Single GPU Training](#single-gpu-training)
  - [Distributed Training](#distributed-training)
- [Roadmap](#roadmap)
  - [Environment](#environment)
  - [Training](#training-1)
  - [Inference](#inference)
  - [Documentation](#documentation)
- [Sponsors](#sponsors)
- [Related Projects](#related-projects)


## Subprojects

### ygoenv
`ygoenv` is a high performance game environment for Yu-Gi-Oh!, implemented on top of [envpool](https://github.com/sail-sg/envpool) and [ygopro-core](https://github.com/Fluorohydride/ygopro-core). It provides standard gym interface for reinforcement learning.

### ygoai
`ygoai` is a set of AI agents for playing Yu-Gi-Oh! It aims to achieve superhuman performance like AlphaGo and AlphaZero, with or without human knowledge. Currently, we focus on using reinforcement learning to train the agents.

## Installation

Pre-built binaries are available for Ubuntu 22.04 or newer. If you're using them, follow the installation instructions below. Otherwise, please build from source following [Building from source](#building-from-source).

1. Install JAX and other dependencies:
   ```bash
   # Install JAX (CPU version)
   pip install -U "jax<=0.4.28"
   # Or with CUDA support
   pip install -U "jax[cuda12]<=0.4.28"

   # Install other dependencies
   pip install flax distrax chex
   ```

2. Clone the repository and install pre-built binary (Ubuntu 22.04 or newer):
   ```bash
   git clone https://github.com/sbl1996/ygo-agent.git
   cd ygo-agent
   # Choose the appropriate version for your Python (cp310, cp311, or cp312)
   wget -nv https://github.com/sbl1996/ygo-agent/releases/download/v0.1/ygopro_ygoenv_cp310.so
   mv ygopro_ygoenv_cp310.so ygoenv/ygoenv/ygopro/ygopro_ygoenv.so
   make
   ```

3. Verify the installation:
   ```bash
   cd scripts
   python -u eval.py --env-id "YGOPro-v1" --deck ../assets/deck/  --num_episodes 32 --strategy random  --lang chinese --num_envs 16
   ```
   If you see episode logs and the output contains this line, the environment is working correctly. For more usage examples, see the [Evaluation](#evaluation) section.

    ```
    len=76.5758, reward=-0.1751, win_rate=0.3939, win_reason=0.9697 
    ```

### Building from source
If you can't use the pre-built binary or prefer to build from source, follow these instructions. Note: These instructions are tested on Ubuntu 22.04 and may not work on other platforms.

#### Additional Prerequisites
- gcc 10+ or clang 11+
- CMake 3.12+
- [xmake](https://xmake.io/#/getting_started)

#### Build Instructions
```bash
git clone https://github.com/sbl1996/ygo-agent.git
cd ygo-agent
xmake f -y
make dev
```

### Troubleshooting

#### Package version not found by xmake
Delete `repositories`, `cache`, `packages` directories in the `~/.xmake` directory and run `xmake f -y -c` again.

#### Install packages failed with xmake
If xmake fails to install required libraries automatically (e.g., `glog` and `gflags`), install them manually (e.g., `apt install`) and add them to the search path (`$LD_LIBRARY_PATH` or others).

#### GLIBC and GLIBCXX version conflict
Mostly, it is because your `libstdc++` from `$CONDA_PREFIX` is older than the system one, while xmake compiles libraries with the system one and you run programs with the `$CONDA_PREFIX` one. If so, you can delete the old `libstdc++` from `$CONDA_PREFIX` (backup it first) and make a soft link to the system one.

#### Other issues
Open a new terminal and try again. If issues persist, join our [Discord channel](https://discord.gg/EqWYj4G4Ys) for help.


## Evaluation

### Obtain a trained agent

We provide trained agents in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1). Check these Flax checkpoint files named with `{exp_id}_{step}.flax_model` and download (the lastest) one to your local machine. The following usage assumes you have it.

### Play against the agent

We can play against the agent with any YGOPro clients now. TODO.

### Battle between two agents

We can use `battle.py` to let two agents play against each other and find out which one is better. Adding `--xla_device cpu` forces JAX to run on CPU.

```bash
python -u battle.py --xla_device cpu --checkpoint1 checkpoints/0546_16500M.flax_model --checkpoint2 checkpoints/0546_11300M.flax_model --num-episodes 32 --seed 0
```

We can set `--record` to generate `.yrp` replay files to the `replay` directory. The `yrp` files can be replayed in YGOPro compatible clients (YGOPro, YGOPro2, KoishiPro, MDPro). Change `--seed` to generate different games.

```bash
python -u battle.py --xla_device cpu --checkpoint1 checkpoints/0546_16500M.flax_model --checkpoint2 checkpoints/0546_11300M.flax_model --num-episodes 16 --seed 1 --record
```

## Training

Training an agent requires a lot of computational resources, typically 8x4090 GPUs and 128-core CPU for a few days. We don't recommend training the agent on your local machine. Reducing the number of decks for training may reduce the computational resources required.

### Single GPU Training

We can train the agent with a single GPU using the following command:

```bash
cd scripts
python -u cleanba.py --actor-device-ids 0 --learner-device-ids 0 \
--local-num_envs 16 --num-minibatches 8 --learning-rate 1e-4 --vloss_clip 1.0 \
--save_interval 100 --local_eval_episodes 32 --eval_interval 50 --seed 0
```

#### Deck
`deck` can be a directory containing `.ydk` files or a single `.ydk` file (e.g., `deck/` or `deck/BlueEyes.ydk`). The well tested and supported decks are in the `assets/deck` directory.

Supported cards are listed in `scripts/code_list.txt`. New decks which only contain supported cards can be used, but errors may also occur due to the complexity of the game.

#### Embedding
To handle the diverse and complex card effects, we have converted the card information and effects into text and used large language models (LLM) to generate embeddings from the text. The embeddings are stored in a file (e.g., `embed.pkl`).

We provide one in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1), which named `embed{n}.pkl` where `n` is the number of cards in `code_list.txt`.

You can choose to not use the embeddings by skip the `--embedding_file` option.

#### Seed
The `seed` option is used to set the random seed for reproducibility. The training and and evaluation will be exactly the same under the same seed.

#### Hyperparameters
More hyperparameters can be found in the `cleanba.py` script. Tuning them may improve the performance but requires more computational resources.

### Distributed Training
TODO

## Roadmap

### Environment
- Generation of yrpX replay files
- Support EDOPro

### Training
- League training (AlphaStar, ROA-Star)
- Nash equilibrium training (OSFP, DeepNash)
- Individual agent for first and second player
- Centralized critic with full observation

### Inference
- Export as SavedModel
- MCTS-based planning

### Documentation
- JAX training
- Custom cards


## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).


## Related Projects
- [ygopro-core](https://github.com/Fluorohydride/ygopro-core)
- [envpool](https://github.com/sail-sg/envpool)
- [neos-ts](https://github.com/DarkNeos/neos-ts)
- [yugioh-ai](https://github.com/melvinzhang/yugioh-ai)
- [yugioh-game](https://github.com/tspivey/yugioh-game)