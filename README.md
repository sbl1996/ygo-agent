# YGO Agent

YGO Agent is a project to create a Yu-Gi-Oh! AI using deep learning (LLMs, RL). It consists of a game environment and a set of AI agents.

## ygoenv
`ygoenv` is a high performance game environment for Yu-Gi-Oh! It is initially inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]) and [yugioh-game](https://github.com/tspivey/yugioh-game), and now implemented on top of [envpool](https://github.com/sail-sg/envpool).

## ygoai
`ygoai` is a set of AI agents for playing Yu-Gi-Oh! It aims to achieve superhuman performance like AlphaGo and AlphaZero, with or without human knowledge. Currently, we focus on using reinforcement learning to train the agents.

## TODO

### Documentation

- Add documentations of building and running

### Training
- Eval with old models during training
- MTCS-based training

### Inference
- MCTS-based planning
- Support of play in YGOPro


## Usage

### Serialize agent

After training, we can serialize the trained agent model to a file for later use without keeping source code of the model. The serialized model file will end with `.ptj` (PyTorch JIT) extension.

```bash
python -u eval.py --agent --checkpoint checkpoints/1234_1000M.pt --num_embeddings 999 --convert --optimize
```

### Battle between two agents

We can use `battle.py` to let two agents play against each other and find out which one is better.

```bash
python -u battle.py --deck ../assets/deck --checkpoint1 checkpoints/1234_1000M.ptj --checkpoint2 checkpoints/9876_100M.ptj --num-episodes=256 --num_envs=32 --seed 0
```

### Running
TODO


## Related Projects
TODO