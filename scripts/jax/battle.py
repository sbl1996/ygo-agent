import sys
import time
import os
import random
from typing import Optional, Literal
from dataclasses import dataclass

import ygoenv
import numpy as np

import tyro

import jax
import jax.numpy as jnp
import flax

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.jax.agent2 import PPOLSTMAgent


@dataclass
class Args:
    seed: int = 1
    """the random seed"""

    env_id: str = "YGOPro-v0"
    """the id of the environment"""
    deck: str = "../assets/deck"
    """the deck file to use"""
    deck1: Optional[str] = None
    """the deck file for the first player"""
    deck2: Optional[str] = None
    """the deck file for the second player"""
    code_list_file: str = "code_list.txt"
    """the code list file for card embeddings"""
    lang: str = "english"
    """the language to use"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 32
    """the number of history actions to use"""
    num_embeddings: Optional[int] = None
    """the number of embeddings of the agent"""

    record: bool = False
    """whether to record the game as YGOPro replays"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""
    verbose: bool = False
    """whether to print debug information"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    rnn_channels: Optional[int] = 512
    """the number of rnn channels for the agent"""
    checkpoint1: str = "checkpoints/agent.pt"
    """the checkpoint to load for the first agent, `pt` or `flax_model` file"""
    checkpoint2: str = "checkpoints/agent.pt"
    """the checkpoint to load for the second agent, `pt` or `flax_model` file"""
    
    # Jax specific
    xla_device: Optional[str] = None
    """the XLA device to use, defaults to `None`"""

    # PyTorch specific
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    compile: bool = False
    """if toggled, the model will be compiled"""
    optimize: bool = False
    """if toggled, the model will be optimized"""
    torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2) * world_size"""

    env_threads: Optional[int] = 16
    """the number of threads to use for envpool, defaults to `num_envs`"""

    framework: Optional[Literal["torch", "jax"]] = None


def create_agent(args):
    return PPOLSTMAgent(
        channels=args.num_channels,
        num_layers=args.num_layers,
        lstm_channels=args.rnn_channels,
        embedding_shape=args.num_embeddings,
    )


def init_rnn_state(num_envs, rnn_channels):
    return (
        np.zeros((num_envs, rnn_channels)),
        np.zeros((num_envs, rnn_channels)),
    )


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

    args = tyro.cli(Args)

    if args.record:
        assert args.num_envs == 1, "Recording only works with a single environment"
        assert args.verbose, "Recording only works with verbose mode"
        if not os.path.exists("replay"):
            os.makedirs("replay")

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)

    deck = init_ygopro(args.env_id, args.lang, args.deck, args.code_list_file)

    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    if args.xla_device is not None:
        os.environ.setdefault("JAX_PLATFORMS", args.xla_device)

    num_envs = args.num_envs

    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=args.env_threads,
        seed=seed,
        deck1=args.deck1,
        deck2=args.deck2,
        player=-1,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        play_mode='self',
        async_reset=False,
        verbose=args.verbose,
        record=args.record,
    )
    obs_space = envs.observation_space
    envs.num_envs = num_envs
    envs = RecordEpisodeStatistics(envs)

    agent = create_agent(args)
    key = jax.random.PRNGKey(args.seed)
    key, agent_key = jax.random.split(key, 2)
    sample_obs = jax.tree_map(lambda x: jnp.array([x]), obs_space.sample())

    rstate = init_rnn_state(1, args.rnn_channels)
    params = jax.jit(agent.init)(agent_key, (rstate, sample_obs))

    with open(args.checkpoint1, "rb") as f:
        params1 = flax.serialization.from_bytes(params, f.read())
    if args.checkpoint1 == args.checkpoint2:
        params2 = params1
    else:
        with open(args.checkpoint2, "rb") as f:
            params2 = flax.serialization.from_bytes(params, f.read())

    @jax.jit
    def get_probs(params, rstate, obs, done):
        agent = create_agent(args)
        next_rstate, logits = agent.apply(params, (rstate, obs))[:2]
        probs = jax.nn.softmax(logits, axis=-1)
        next_rstate = jax.tree_map(
            lambda x: jnp.where(done[:, None], 0, x), next_rstate)
        return next_rstate, probs

    if args.num_envs != 1:
        @jax.jit
        def get_probs2(params1, params2, rstate1, rstate2, obs, main, done):
            next_rstate1, probs1 = get_probs(params1, rstate1, obs, done)
            next_rstate2, probs2 = get_probs(params2, rstate2, obs, done)
            probs = jnp.where(main[:, None], probs1, probs2)
            rstate1 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x1, x2), next_rstate1, rstate1)
            rstate2 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x2, x1), next_rstate2, rstate2)
            return rstate1, rstate2, probs

        def predict_fn(rstate1, rstate2, obs, main, done):
            rstate1, rstate2, probs = get_probs2(params1, params2, rstate1, rstate2, obs, main, done)
            return rstate1, rstate2, np.array(probs)
    else:
        def predict_fn(rstate1, rstate2, obs, main, done):
            if main[0]:
                rstate1, probs = get_probs(params1, rstate1, obs, done)
            else:
                rstate2, probs = get_probs(params2, rstate2, obs, done)
            return rstate1, rstate2, np.array(probs)

    obs, infos = envs.reset()
    next_to_play = infos['to_play']
    dones = np.zeros(num_envs, dtype=np.bool_)

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    win_reasons = []

    step = 0
    start = time.time()
    start_step = step

    main_player = np.concatenate([
        np.zeros(num_envs // 2, dtype=np.int64),
        np.ones(num_envs - num_envs // 2, dtype=np.int64)
    ])
    rstate1 = rstate2 = init_rnn_state(num_envs, args.rnn_channels)

    model_time = env_time = 0
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        _start = time.time()
        main = next_to_play == main_player
        rstate1, rstate2, probs = predict_fn(rstate1, rstate2, obs, main, dones)

        actions = probs.argmax(axis=1)
        model_time += time.time() - _start

        to_play = next_to_play

        _start = time.time()
        obs, rewards, dones, infos = envs.step(actions)
        next_to_play = infos['to_play']
        env_time += time.time() - _start

        step += 1

        for idx, d in enumerate(dones):
            if d:
                win_reason = infos['win_reason'][idx]
                pl = 1 if to_play[idx] == main_player[idx] else -1
                episode_length = infos['l'][idx]
                episode_reward = infos['r'][idx] * pl
                win = int(episode_reward > 0)

                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                win_rates.append(win)
                win_reasons.append(1 if win_reason == 1 else 0)
                sys.stderr.write(f"Episode {len(episode_lengths)}: length={episode_length}, reward={episode_reward}, win={win}, win_reason={win_reason}\n")

                # Only when num_envs=1, we switch the player here
                if args.verbose:
                    main_player = 1 - main_player

        if len(episode_lengths) >= args.num_episodes:
            break

    print(f"len={np.mean(episode_lengths)}, reward={np.mean(episode_rewards)}, win_rate={np.mean(win_rates)}, win_reason={np.mean(win_reasons)}")

    total_time = time.time() - start
    total_steps = (step - start_step) * num_envs
    print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
    print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
    