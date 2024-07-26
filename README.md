# YGO Agent

YGO Agent is a project aimed at mastering the popular trading card game Yu-Gi-Oh! through deep learning. Based on a high-performance game environment (ygoenv), this project leverages reinforcement learning and large language models to develop advanced AI agents (ygoai) that aim to match or surpass human expert play. YGO Agent provides researchers and players with a platform for exploring AI in complex, strategic game environments.

[Discord](https://discord.gg/EqWYj4G4Ys)

## News🔥

- 2024.7.17 - We have launched the human-AI battle feature in [Neos](https://neos.moecube.com/). Check the [Single GPU Training](#single-gpu-training) and [Play against the agent](#play-against-the-agent) sections to train your own model and play against it in your favorite YGOPro clients!
- 2024.7.2 - We have a discord channel for discussion now! We are also working with [neos-ts](https://github.com/DarkNeos/neos-ts) to implement human-AI battle.
- 2024.4.18 - LSTM has been implemented and well tested.
- 2024.4.7 - We have switched to JAX for training and evaluation due to the better performance and flexibility.


## Table of Contents
- [Subprojects](#subprojects)
  - [ygoenv](#ygoenv)
  - [ygoai](#ygoai)
- [Installation](#installation)
  - [Quick start](#quick-start)
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

### Quick start
Pre-built binaries are available for Ubuntu 22.04 or newer. If you're using them, follow the installation instructions below. Otherwise, please build from source following [Building from source](#building-from-source).

1. Install JAX and other dependencies:
   ```bash
   # Install JAX (CPU version)
   pip install -U "jax<=0.4.28" "jaxlib<=0.4.28"
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
   wget -O ygopro_ygoenv.so https://github.com/sbl1996/ygo-agent/releases/download/v0.1/ygopro_ygoenv_cp310.so
   mv ygopro_ygoenv.so ygoenv/ygoenv/ygopro/ygopro_ygoenv.so
   make
   ```

3. Verify the installation:
   ```bash
   cd scripts
   python -u eval.py --deck ../assets/deck/  --num_episodes 32 --strategy random  --num_envs 16
   ```
   If you see episode logs and the output contains this line, the environment is working correctly. For more usage examples, see the [Evaluation](#evaluation) section.

   ```
   len=76.5758, reward=-0.1751, win_rate=0.3939, win_reason=0.9697 
   ```
   The most common problem you might encounter is the GLIBCXX version issue, such as:
   ```
   ImportError: /home/hastur/miniconda3/envs/ygo/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /home/hastur/Code/ygo-agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.so)
   ```
   Please refer to [GLIBC and GLIBCXX version conflict](#glibc-and-glibcxx-version-conflict) for solution.


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
Mostly, it is because your `libstdc++` from `$CONDA_PREFIX` is older than the system one, while xmake compiles libraries with the system one and you run programs with the `$CONDA_PREFIX` one. If so, you can delete the old `libstdc++` from `$CONDA_PREFIX` (backup it first) and make a soft link to the system one. You may refer to the following step:

```bash
cd $CONDA_PREFIX/lib
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 libstdc++.so.6.0.30
rm libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6.0.30 libstdc++.so.6
ln -s libstdc++.so.6 libstdc++.so
```

#### Other issues
Open a new terminal and try again. If issues persist, join our [Discord channel](https://discord.gg/EqWYj4G4Ys) for help.


## Evaluation

### Obtain a trained agent

We provide trained agents in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1). Check these checkpoint files named with `{exp_id}_{step}.(flax_model|tflite)` and download (the lastest) one to your local machine. The following usage assumes you have it.

### Play against the agent

We can play against the agent with any YGOPro clients now!

#### Deploy model as API service

```bash
cd scripts
export CHECKPOINT=checkpoints/0546_26550M.tflite
uvicorn ygoinf.server:app --host 127.0.0.1 --port 3000 --log-config=../assets/log_conf.yaml
```

Run this command to deploy the model as an API service, which compatible clients can use to implement AI feature. On WSL2, the port will be automatically mapped to local; if not, it's recommended to use VSCode's port forwarding feature.

After setting up, try accessing http://127.0.0.1:3000 in your browser to see if it opens. If normal, it will display `OK`, indicating successful deployment. If not, try changing the port and trying again.

#### Play through Neos

[Neos](https://github.com/DarkNeos/neos-ts) is an open-source YGOPro web client, which is very convenient to use — you can play just by opening a browser. We've collaborated with the Neos team to launch client AI feature based on the model API service.

Let's open [Neos](https://neos.moecube.com/) and click "Start game" to register and log in to a Moecube account. Then add your deck in "Deck Building". Supported decks are list [here](./assets/deck/).

Then go to "Match", move your mouse to the avatar in the upper right corner, click on "System Settings" that appears, click "AI Settings", enter "http://127.0.0.1:3000" in the box, click "Apply AI Settings" to save, and click anywhere else to close the system settings.

After that, click "Custom Room", turn on the "Enable AI Assist" option, enter a player nickname, enter "NOCHECK#XXXXXX" for the room password, where XXXXXX is any 6-10 digit number. Make it complex to avoid duplication with others. Remember this password as you'll need it later. Then click "Join Room", select deck in the "Deck", and then click "Duel Ready".

Next, if you have other YGOPro clients, you can join this room on the Koishi server (http://koishi.momobako.com:7210) through them. If you don't have other clients, press "Ctrl+N" to open a new browser window, open Neos again, select "Custom Room", don't turn on AI assist, join this room using the password you entered earlier, also select deck, and click "Duel Ready". Then go back to the previous Neos window, click "Start Game". Start a fun duel with the AI!

Note that Neos must remain in the foreground, otherwise the game will pause. You can keep the other YGOPro client or the other Neos browser client window in front of the Neos window running the AI.


### Battle between two agents

We can use `battle.py` to let two agents play against each other and find out which one is better. Adding `--xla_device cpu` forces JAX to run on CPU.

```bash
python -u battle.py --xla_device cpu --checkpoint1 checkpoints/0546_22750M.flax_model --checkpoint2 checkpoints/0546_11300M.flax_model --num-episodes 32 --seed 0
```

We can set `--record` to generate `.yrp` replay files to the `replay` directory. The `yrp` files can be replayed in YGOPro compatible clients (YGOPro, YGOPro2, KoishiPro, MDPro). Change `--seed` to generate different games.

```bash
python -u battle.py --xla_device cpu --checkpoint1 checkpoints/0546_22750M.flax_model --checkpoint2 checkpoints/0546_11300M.flax_model --num-episodes 16 --seed 1 --record
```

## Training

### Single GPU Training

The minimum requirement of training is a NVIDIA GPU. I can even train on a laptop with a GeForce GTX 1650. The supported decks that can be found [here](./assets/deck/). Any combination of cards included in these decks is fine, and more will be added later. For demonstration, we'll choose just one deck to train from scratch.

```bash
cd scripts
python -u cleanba.py --actor-device-ids 0 --learner-device-ids 0 --deck ../assets/deck/BlueEyes.ydk \
--local-num_envs 16 --num-minibatches 8 --learning-rate 1e-4 --vloss_clip 1.0 \
--save_interval 50 --local_eval_episodes 32 --eval_interval 50 --seed 0 --tb_dir None --checkpoint checkpoints/XXX.flax_model
```

We specify the location of the Blue-Eyes White Dragon deck to be trained through `--deck`. The training then opens `16` parallel environments on each actor, with 2 actors by default, for a total of 32 parallel environments. Every 128 steps is one exploration training cycle (iter), and all samples obtained are divided into `8` minibatches, resulting in a minibatch size of 512, corresponding to a learning rate of `1e-4`. `--save_interval` indicates saving the model every 50 iters, and `--eval_interval` indicates evaluating against a random strategy every 50 iters.

```
obs_space=Dict('cards_': Box(0, 255, (160, 41), uint8), 'global_': Box(0, 255, (23,), uint8), 'actions_': Box(0, 255, (24, 12), uint8), 'h_actions_': Box(0, 255, (32, 14), uint8), 'mask_': Box(0, 255, (160, 14), uint8)), action_shape=()
global_step=40960, avg_return=0.1143, avg_length=313
16:28:06 SPS: 965, update: 965, rollout_time=2.18, params_time=1.72
40960 actor_update=10, train_time=3.98, data_time=0.00, put_time=0.00
global_step=81920, avg_return=0.0850, avg_length=246
16:28:47 SPS: 1012, update: 998, rollout_time=2.31, params_time=1.51
81920 actor_update=20, train_time=4.04, data_time=0.00, put_time=0.00
global_step=122880, avg_return=0.0694, avg_length=189
16:29:27 SPS: 1013, update: 1123, rollout_time=2.17, params_time=1.10
122880 actor_update=30, train_time=4.05, data_time=0.00, put_time=0.00
global_step=163840, avg_return=0.1003, avg_length=192
16:30:07 SPS: 1016, update: 1024, rollout_time=2.24, params_time=1.69
163840 actor_update=40, train_time=4.02, data_time=0.00, put_time=0.00
global_step=204800, avg_return=0.0676, avg_length=194
16:30:59 SPS: 1014, update: 261, rollout_time=2.29, params_time=1.47
eval_time=11.6271, eval_return=1.4659, eval_win_rate=0.9844
204800 actor_update=50, train_time=3.97, data_time=10.07, put_time=0.00
Saved model to /home/hastur/Code/ygo-agent/scripts/checkpoints/1720859207_0M.flax_model
```

I trained for 50 iters with the results shown above. On my laptop with a 1650, it can train 1000 steps per second. I believe many of you have much better GPU, so the training speed will be much faster. From the `eval_win_rate`, we can see that after just 50 iters, it's nearly 100% defeating the random strategy. Let's now compare it with another model ([download](https://github.com/sbl1996/ygo-agent/releases/download/v0.1/0546_22750M.flax_model)) I trained for over 100M games (equivalent to training on 32 4090 GPUs for 5 days):

```bash
> python -u battle.py --num_episodes 128 --deck ../assets/deck/BlueEyes.ydk  --seed 0 \
  --checkpoint1 checkpoints/1720859207_0M.flax_model  --checkpoint2 checkpoints/0546_22750M.flax_model

 100%|██████████████████████████████████████████████████████████████████| 128/128 [00:28<00:00,  4.44it/s, len=144, reward=-2.5, win_rate=0.0312]
len=143.6796875, reward=-2.5041699409484863, win_rate=0.03125, win_reason=1.0
Payoff matrix:
   agent1  agent2
0  0.0078  0.4766
1  0.0234  0.4922
0/1 matrix, win rates of agentX as playerY
   agent1  agent2
0  0.0156  0.9531
1  0.0469  0.9844
Length matrix, length of games of agentX as playerY
   agent1  agent2
0  139.97  147.39
1  147.39  139.97
SPS: 2478, total_steps: 39936
total: 16.1189, model: 13.2234, env: 2.8318
```

Out of 128 games, the total win rate is 3.125%, with a 1.56% win rate going first and 4.69% going second. After all, the difference in training steps is nearly 60,000 times. You can try training your own model and use the win rate against this model as the main metric to improve your training methods.

Now, you may also check the [Play against the agent](#play-against-the-agent) section to play against your trained model with your favorite YGOPro clients.

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
Training an agent with many decks requires a lot of computational resources, typically 8x4090 GPUs and 128-core CPU for a few days. Therefore, we need distributed training.

TODO

## Roadmap

### Game play
- Client-side deployment like Windbot

### Environment
- Support more cards (first windbot and top tier decks)
- Generation of yrpX and yrp3d replay files
- Support EDOPro

### YGO
- Implement AI deck building
- Support BO3, able to exploit information from previous duels

### RL
- Fast fine-tuning methods for new decks
- Zero-shot generalization of new cards
- MCTS Planning
- Nash equilibrium and League training


## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).


## Related Projects
- [ygopro-core](https://github.com/Fluorohydride/ygopro-core)
- [envpool](https://github.com/sail-sg/envpool)
- [neos-ts](https://github.com/DarkNeos/neos-ts)
- [yugioh-ai](https://github.com/melvinzhang/yugioh-ai)
- [yugioh-game](https://github.com/tspivey/yugioh-game)