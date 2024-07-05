# YGO Agent

YGO Agent is a project to create a Yu-Gi-Oh! AI using deep learning (LLMs, RL). It consists of a game environment and a set of AI agents.

[Discord](https://discord.gg/EqWYj4G4Ys)

## News

- July 2, 2024: We have a discord channel for discussion now! We are also working with [neos-ts](https://github.com/DarkNeos/neos-ts) to implement human-AI battle.

- April 18, 2024: We have fully switched to JAX for training and evaluation. Check the evaluation sections for more details and try the new JAX-trained agents.

- April 14, 2024: LSTM has been implemented and well tested. See `scripts/jax/ppo.py` for more details.

- April 7, 2024: We have switched to JAX for training and evalution due to the better performance and flexibility. The scripts are in the `scripts/jax` directory. The documentation is in progress. PyTorch scripts are still available in the `scripts` directory, but they are not maintained.


## Table of Contents
- [Subprojects](#subprojects)
  - [ygoenv](#ygoenv)
  - [ygoai](#ygoai)
- [Building](#building)
  - [Common Issues](#common-issues)
- [Evaluation](#evaluation)
  - [Obtain a trained agent](#obtain-a-trained-agent)
  - [Play against the agent](#play-against-the-agent)
  - [Battle between two agents](#battle-between-two-agents)
- [Training (Deprecated, to be updated)](#training-deprecated-to-be-updated)
  - [Single GPU Training](#single-gpu-training)
  - [Distributed Training](#distributed-training)
- [Plan](#plan)
  - [Environment](#environment)
  - [Training](#training-1)
  - [Inference](#inference)
  - [Documentation](#documentation)
- [Sponsors](#sponsors)
- [Related Projects](#related-projects)


## Subprojects

### ygoenv
`ygoenv` is a high performance game environment for Yu-Gi-Oh! It is initially inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]) and [yugioh-game](https://github.com/tspivey/yugioh-game), and now implemented on top of [envpool](https://github.com/sail-sg/envpool).

### ygoai
`ygoai` is a set of AI agents for playing Yu-Gi-Oh! It aims to achieve superhuman performance like AlphaGo and AlphaZero, with or without human knowledge. Currently, we focus on using reinforcement learning to train the agents.


## Building

The following building instructions are only tested on Ubuntu (WSL2) and may not work on other platforms.

To build the project, you need to install the following prerequisites first:
- gcc 10+ or clang 11+
- CMake 3.12+
- [xmake](https://xmake.io/#/getting_started)
- jax 0.4.25+, flax 0.8.2+, distrax 0.1.5+ (CUDA is optional)

After that, you can build with the following commands:

```bash
git clone https://github.com/sbl1996/ygo-agent.git
cd ygo-agent
git checkout stable  # switch to the stable branch
xmake f -y
make
```

After building, you can run the following command to test the environment. If you see episode logs, it means the environment is working. Try more usage in the next section!

```bash
cd scripts
python -u eval.py --env-id "YGOPro-v1" --deck ../assets/deck/  --num_episodes 32 --strategy random  --lang chinese --num_envs 16
```

### Common Issues

#### Package version not found by xmake
Delete `repositories`, `cache`, `packages` directories in the `~/.xmake` directory and run `xmake f -y` again.

#### Install packages failed with xmake
Sometimes you may fail to install the required libraries by xmake automatically (e.g., `glog` and `gflags`). You can install them manually (e.g., `apt install`) and put them in the search path (`$LD_LIBRARY_PATH` or others), then xmake will find them.

#### GLIBC and GLIBCXX version conflict
Mostly, it is because your `libstdc++` from `$CONDA_PREFIX` is older than the system one, while xmake compiles libraries with the system one and you run programs with the `$CONDA_PREFIX` one. If so, you can delete the old `libstdc++` from `$CONDA_PREFIX` (backup it first) and make a soft link to the system one.

#### Other issues
Open a new terminal and try again. If you still encounter issues, you can join the [Discord channel](https://discord.gg/EqWYj4G4Ys) for help.


## Evaluation

### Obtain a trained agent

We provide trained agents in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1). Check these Flax checkpoint files named with `{commit_hash}_{exp_id}_{step}.flax_model` and download (the lastest) one to your local machine. The following usage assumes you have it.

If you are not in the `stable` branch or encounter any other running issues, you can try to switch to the `commit_hash` commit before using the agent. You may need to rebuild the project after switching:

```bash
xmake f -c
xmake b -r ygopro_ygoenv
```

### Play against the agent

We can use `eval.py` to play against the trained agent with a MUD-like interface in the terminal. We add `--xla_device cpu` to run the agent on the CPU.

```bash
python -u eval.py --deck ../assets/deck --lang chinese --xla_device cpu --checkpoint checkpoints/350c29a_7565_6700M.flax_model --play
```

We can enter `quit` to exit the game. Run `python eval.py --help` for more options, for example, `--player 0` to make the agent play as the first player, `--deck1 TenyiSword` to force the first player to use the TenyiSword deck.


### Battle between two agents

We can use `battle.py` to let two agents play against each other and find out which one is better.

```bash
python -u battle.py --deck ../assets/deck --checkpoint1 checkpoints/350c29a_7565_6700M.flax_model --checkpoint2 checkpoints/350c29a_1166_6200M.flax_model --num-episodes 32 --num_envs 8 --seed 0
```

We can set `--record` to generate `.yrp` replay files to the `replay` directory. The `yrp` files can be replayed in YGOPro compatible clients (YGOPro, YGOPro2, KoishiPro, MDPro). Change `--seed` to generate different games.

```bash
python -u battle.py --deck ../assets/deck --xla_device cpu --checkpoint1 checkpoints/350c29a_7565_6700M.flax_model --checkpoint2 checkpoints/350c29a_1166_6200M.flax_model --num-episodes 16 --record --seed 0
```


## Training

Training an agent requires a lot of computational resources, typically 8x4090 GPUs and 128-core CPU for a few days. We don't recommend training the agent on your local machine. Reducing the number of decks for training may reduce the computational resources required.

### Single GPU Training

We can train the agent with a single GPU using the following command:

```bash
cd scripts
python -u cleanba.py --actor-device-ids 0 --learner-device-ids 0 \
--local-num_envs 16 --num-minibatches 8 --learning-rate 1e-4 \
--update-epochs 1 --vloss_clip 1.0 --sep_value --value gae \
--save_interval 100 --seed 0 --m1.film --m1.noam --m1.version 2 \
--local_eval_episodes 32 --eval_interval 50
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

## Plan

### Environment
- Generation of yrpX replay files
- Support EDOPro

### Training
- League training (AlphaStar, ROA-Star)
- Nash equilibrium training (OSFP, DeepNash)
- Individual agent for first and second player
- Centralized critic with full observation

### Inference
- MCTS-based planning
- Support of play in YGOPro

### Documentation
- JAX training
- Custom cards


## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).


## Related Projects
- [ygopro-core](https://github.com/Fluorohydride/ygopro-core)
- [envpool](https://github.com/sail-sg/envpool)
- [yugioh-ai](https://github.com/melvinzhang/yugioh-ai)
- [yugioh-game](https://github.com/tspivey/yugioh-game)
