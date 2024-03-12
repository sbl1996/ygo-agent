# YGO Agent

YGO Agent is a project to create a Yu-Gi-Oh! AI using deep learning (LLMs, RL). It consists of a game environment and a set of AI agents.

## ygoenv
`ygoenv` is a high performance game environment for Yu-Gi-Oh! It is initially inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]) and [yugioh-game](https://github.com/tspivey/yugioh-game), and now implemented on top of [envpool](https://github.com/sail-sg/envpool).

## ygoai
`ygoai` is a set of AI agents for playing Yu-Gi-Oh! It aims to achieve superhuman performance like AlphaGo and AlphaZero, with or without human knowledge. Currently, we focus on using reinforcement learning to train the agents.


## Building

### Prerequisites
- gcc 10+ or clang 11+
- [xmake](https://xmake.io/#/getting_started)
- PyTorch 2.0 or later with cuda support

After installing the prerequisites, you can build the project with the following commands:

```bash
git clone https://github.com/sbl1996/ygo-agent.git
cd ygo-agent
git checkout eval_with_ptj  # checkout to the stable branch
xmake f -y
make
```

Sometimes you may fail to install the required libraries by xmake automatically (e.g., `glog` and `gflags`). You can install them manually and put them in the search path (LD_LIBRARY_PATH or others), then xmake will find them.

After building, you can run the following command to test the environment. If you see episode logs, it means the environment is working. Try more usage in the next section!
```bash
cd scripts
python -u eval.py --env-id "YGOPro-v0" --deck ../assets/deck/  --num_episodes 32 --strategy random  --lang chinese --num_envs 16
```


## Usage

### Obtain a trained agent

We provide some trained agents in the [releases](https://github.com/sbl1996/ygo-agent/releases/tag/v0.1). Check these `ptj` TorchScript files and download them to your local machine. The following usage assumes you have it.

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

## TODO

### Training
- Evaluation with old models during training
- LSTM for memory
- League training following AlphaStar and ROA-Star

### Inference
- MCTS-based planning
- Support of play in YGOPro

### Related Projects
- [yugioh-ai](https://github.com/melvinzhang/yugioh-ai])
- [yugioh-game](https://github.com/tspivey/yugioh-game)
- [envpool](https://github.com/sail-sg/envpool)