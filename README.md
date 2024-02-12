# Yugioh AI

Yugioh AI uses large language models (LLM) and RL to play Yu-Gi-Oh. It is inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]) and [yugioh-game](https://github.com/tspivey/yugioh-game), and uses [ygopro-core](https://github.com/Fluorohydride/ygopro-core).

## Usage

### Setup

An automated setup script is provided for Linux (tested on Ubuntu 22.04). It will install all dependencies and build the library. To run it, execute the following commands:

```bash
make setup
```

### Running

Test that the repo is setup correctly by running:

```
python cli.py --deck1 deck/Starter.ydk --deck2 deck/BlueEyes.ydk
```

You should see text output showing two random AI playing a duel by making random moves.

You can set `--seed` to a fixed value to get deterministic results.

## Implementation

The implementation is initially based on [yugioh-game](https://github.com/tspivey/yugioh-game). To provide a clean game environment, it removes all server code and only keeps basic duel-related classes and message handlers.
To implement the AI, inspired by [yugioh-ai](https://github.com/melvinzhang/yugioh-ai]), every message handler also provides all possible actions that can be taken in response to the message.

## Notes
Never 