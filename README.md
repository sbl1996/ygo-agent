# YGO Agent

YGO Agent is a project to create a Yu-Gi-Oh! AI using deep learning (LLMs, RL). It consists of a game environment and a set of AI agents.

## ygoenv
`ygoenv` is a high performance game environment for Yu-Gi-Oh! It is initially inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]) and [yugioh-game](https://github.com/tspivey/yugioh-game), and now implemented on top of [envpool](https://github.com/sail-sg/envpool).

## ygoai
`ygoai` is a set of AI agents for playing Yu-Gi-Oh! It aims to achieve superhuman performance like AlphaGo and AlphaZero, with or without human knowledge. Currently, we focus on using reinforcement learning to train the agents.


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

### Documentation

- Add documentations of building and running

### Training
- Evaluation with old models during training
- LSTM for memory
- League training following AlphaStar and ROA-Star

### Inference
- MCTS-based planning
- Support of play in YGOPro


## Related Projects
TODO