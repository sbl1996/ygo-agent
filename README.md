# YGO Agent

YGO Agent is a project to create a Yu-Gi-Oh! AI using deep learning (LLMs, RL). It consists of a game environment and a set of AI agents.

## News

- April 7, 2024: We have switched to JAX for training and evalution due to the better performance and flexibility. The scripts are in the `scripts/jax` directory. The documentation is in progress. PyTorch scripts are still available in the `scripts` directory, but they are not maintained.


## Table of Contents
- [Subprojects](#subprojects)
  - [ygoenv](#ygoenv)
  - [ygoai](#ygoai)
- [Building](#building)
- [Evaluation](#evaluation)
  - [Obtain a trained agent](#obtain-a-trained-agent)
  - [Play against the agent](#play-against-the-agent)
  - [Battle between two agents](#battle-between-two-agents)
  - [Serialize agent](#serialize-agent)
- [Training](#training)
  - [Single GPU Training](#single-gpu-training)
  - [Distributed Training](#distributed-training)
- [Training (JAX)](#training-jax)
- [Plan](#plan)
  - [Training](#training-1)
  - [Inference](#inference)
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
- PyTorch 2.0 or later with cuda support

After that, you can build with the following commands:

```bash
git clone https://github.com/sbl1996/ygo-agent.git
cd ygo-agent
git checkout eval_with_ptj  # switch to the stable branch
xmake f -y
make
```

Sometimes you may fail to install the required libraries by xmake automatically (e.g., `glog` and `gflags`). You can install them manually and put them in the search path (LD_LIBRARY_PATH or others), then xmake will find them.

After building, you can run the following command to test the environment. If you see episode logs, it means the environment is working. Try more usage in the next section!

```bash
cd scripts
python -u eval.py --env-id "YGOPro-v0" --deck ../assets/deck/  --num_episodes 32 --strategy random  --lang chinese --num_envs 16
```

## Evaluation

### Obtain a trained agent

We provide some trained agents in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1). Check these TorchScript files named with `{commit_hash}_{exp_id}_{step}.ptj` and download them to your local machine. Switch to the corresponding commit hash before using it. The following usage assumes you have it.

Notice that the provided `ptj` can only run on GPU, but not CPU. Actually, the agent can run in real-time on CPU, we will provide a CPU version in the future.

### Play against the agent

We can use `eval.py` to play against the trained agent with a MUD-like interface in the terminal.

```bash
python -u eval.py --agent --deck ../assets/deck  --lang chinese --checkpoint checkpoints/1234_1000M.ptj --play
```

### Battle between two agents

We can use `battle.py` to let two agents play against each other and find out which one is better.

```bash
python -u battle.py --deck ../assets/deck --checkpoint1 checkpoints/1234_1000M.ptj --checkpoint2 checkpoints/9876_100M.ptj --num-episodes=256 --num_envs=32 --seed 0
```

You can set `--num_envs=1 --verbose --record` to generate `.yrp` replay files.


### Serialize agent

After training, we can serialize the trained agent model to a file for later use without keeping source code of the model. The serialized model file will end with `.ptj` (PyTorch JIT) extension.

```bash
python -u eval.py --agent --checkpoint checkpoints/1234_1000M.pt --num_embeddings 999 --convert --optimize
```

If you have used `--embedding_file` during training, skip the `--num_embeddings` option.

## Training

Training an agent requires a lot of computational resources, typically 8x4090 GPUs and 128-core CPU for a few days. We don't recommend training the agent on your local machine. Reducing the number of decks for training may reduce the computational resources required.

### Single GPU Training

We can train the agent with a single GPU using the following command:

```bash
python -u ppo.py --deck ../assets/deck --seed 1 --embedding_file embed.pkl \
--minibatch-size 128 --learning-rate 1e-4 --update-epochs 2  --save_interval 100 \
--compile reduce-overhead --env_threads 16 --num_envs 64 --eval_episodes 32
```

#### Deck
`deck` can be a directory containing `.ydk` files or a single `.ydk` file (e.g., `deck/` or `deck/BlueEyes.ydk`). The well tested and supported decks are in the `assets/deck` directory.

Supported cards are listed in `scripts/code_list.txt`. New decks which only contain supported cards can be used, but errors may also occur due to the complexity of the game.

#### Embedding
To handle the diverse and complex card effects, we have converted the card information and effects into text and used large language models (LLM) to generate embeddings from the text. The embeddings are stored in a file (e.g., `embed.pkl`).

We provide one in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1), which named `embed{n}.pkl` where `n` is the number of cards in `code_list.txt`.

You can choose to not use the embeddings by skip the `--embedding_file` option. If you do it, remember to set `--num_embeddings` to `999` in the `eval.py` script.

#### Compile
We use `torch.compile` to speed up the overall training process. It is very important and can reduce the overall time by 2x or more. If the compilation fails, you may update the PyTorch version to the latest one.

#### Seed
The `seed` option is used to set the random seed for reproducibility. However, many optimizations used in the training are not deterministic, so the results may still vary.

For debugging, you can set `--compile None --torch-deterministic` with the same seed to get a deterministic result.

#### Hyperparameters
More PPO hyperparameters can be found in the `ppo.py` script. Tuning them may improve the performance but requires more computational resources.


### Distributed Training

The `ppo.py` script supports single-node and multi-node distributed training with `torchrun`. Start distributed training like this:

```bash
# single node
OMP_NUM_THREADS=4 torchrun --standalone --nnodes=1 --nproc-per-node=8 ppo.py \

# multi node on nodes 0
OMP_NUM_THREADS=4 torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
--rdzv-id=12941 --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT ppo.py \
# multi node on nodes 1
OMP_NUM_THREADS=4 torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
--rdzv-id=12941 --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT ppo.py \


# script options
--deck ../assets/deck --seed 1 --embedding_file embed.pkl \
--minibatch-size 2048 --learning-rate 5e-4 --update-epochs 2 --save_interval 100 \
--compile reduce-overhead --env_threads 128 --num_envs 1024 --eval_episodes 128
```

The script options are mostly the same as the single GPU training. We only scale the batch size and the number of environments to the number of available CPUs and GPUs. The learning rate is then scaled according to the batch size.

## Plan

### Environment
- Fix information leak in the history actions

### Training
- Evaluation with old models during training
- League training following AlphaStar and ROA-Star

### Inference
- MCTS-based planning
- Support of play in YGOPro

### Documentation
- JAX training and evaluation


## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).


## Related Projects
- [ygopro-core](https://github.com/Fluorohydride/ygopro-core)
- [envpool](https://github.com/sail-sg/envpool)
- [yugioh-ai](https://github.com/melvinzhang/yugioh-ai)
- [yugioh-game](https://github.com/tspivey/yugioh-game)
