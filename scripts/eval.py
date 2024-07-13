import sys
import time
import os
import random
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict

import ygoenv
import numpy as np

import tyro

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.jax.agent import RNNAgent, ModelArgs


@dataclass
class Args:
    seed: int = 1
    """the random seed"""

    env_id: str = "YGOPro-v1"
    """the id of the environment"""
    deck: str = "../assets/deck"
    """the deck file to use"""
    deck1: Optional[str] = None
    """the deck name for the first player, for example, `Hero`"""
    deck2: Optional[str] = None
    """the deck name for the second player, for example, `CyberDragon`"""
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

    player: int = -1
    """the player to play as, -1 means random, 0 is the first player, 1 is the second player"""
    play: bool = False
    """whether to play the game"""
    verbose: bool = False
    """whether to print debug information"""
    record: bool = False
    """whether to record the game as YGOPro replays"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""

    bot_type: Literal["random", "greedy"] = "greedy"
    """the type of bot to use"""
    strategy: Literal["random", "greedy"] = "greedy"
    """the strategy to use if agent is not used"""

    m: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent1"""

    checkpoint: Optional[str] = None
    """the checkpoint to load, must be a `flax_model` file"""

    xla_device: Optional[str] = None
    """the XLA device to use, `cpu` for forcing running on CPU"""

    env_threads: Optional[int] = None
    """the number of threads to use for envpool, defaults to `num_envs`"""


def create_agent(args):
    return RNNAgent(
        **asdict(args.m),
        embedding_shape=args.num_embeddings,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.play or args.record:
        args.num_envs = 1
        args.verbose = True
        print("Set num_envs=1 and verbose=True for recording or playing the game")
        if args.record and not os.path.exists("replay"):
            os.makedirs("replay")

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)

    deck, deck_names = init_ygopro(args.env_id, args.lang, args.deck, args.code_list_file, return_deck_names=True)

    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    seed = args.seed + 100000
    random.seed(seed)
    seed = random.randint(0, int(1e8))
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
        player=args.player,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        play_mode='human' if args.play else ('bot' if args.bot_type == "greedy" else "random"),
        async_reset=False,
        verbose=args.verbose,
        record=args.record,
    )
    obs_space = envs.observation_space
    envs.num_envs = num_envs
    envs = RecordEpisodeStatistics(envs)

    if args.checkpoint:
        import jax
        import jax.numpy as jnp
        import flax
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

        agent = create_agent(args)
        key = jax.random.PRNGKey(seed)
        sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())

        rstate = agent.init_rnn_state(1)
        params = jax.jit(agent.init)(key, sample_obs, rstate)

        with open(args.checkpoint, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())

        params = jax.device_put(params)
        rstate = agent.init_rnn_state(num_envs)

        @jax.jit
        def get_probs_and_value(params, rstate, obs, done):
            next_rstate, logits, value = agent.apply(params, obs, rstate)[:3]
            probs = jax.nn.softmax(logits, axis=-1)
            next_rstate = jax.tree.map(
                lambda x: jnp.where(done[:, None], 0, x), next_rstate)
            return next_rstate, probs, value

        def predict_fn(rstate, obs, done):
            rstate, probs, value = get_probs_and_value(params, rstate, obs, done)
            return rstate, np.array(probs), np.array(value)

        print(f"loaded checkpoint from {args.checkpoint}")


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

    deck_names = sorted(deck_names)
    deck_times = {name: 0 for name in deck_names}
    deck_time_count = {name: 0 for name in deck_names}

    model_time = env_time = 0
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        if args.checkpoint:
            _start = time.time()
            rstate, probs, value = predict_fn(rstate, obs, dones)
            if args.verbose:
                print(f"probs: {[f'{p:.4f}' for p in probs[probs != 0].tolist()]}")
                print(f"value: {value[0][0]}")
            actions = probs.argmax(axis=1)
            model_time += time.time() - _start
        else:
            if args.strategy == "random":
                actions = np.random.randint(infos['num_options'])
            else:
                actions = np.zeros(num_envs, dtype=np.int32)

        to_play = next_to_play

        _start = time.time()
        obs, rewards, dones, infos = envs.step(actions)
        next_to_play = infos['to_play']
        env_time += time.time() - _start

        step += 1

        for idx, d in enumerate(dones):
            if not d:
                continue

            win_reason = infos['win_reason'][idx]
            episode_length = infos['l'][idx]
            episode_reward = infos['r'][idx]
            win = int(episode_reward > 0)

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            win_rates.append(win)
            win_reasons.append(1 if win_reason == 1 else 0)
            sys.stderr.write(f"Episode {len(episode_lengths)}: length={episode_length}, reward={episode_reward}, win={win}, win_reason={win_reason}\n")
        if len(episode_lengths) >= args.num_episodes:
            break

    print(f"len={np.mean(episode_lengths):.4f}, reward={np.mean(episode_rewards):.4f}, win_rate={np.mean(win_rates):.4f}, win_reason={np.mean(win_reasons):.4f}")
    if not args.play:
        total_time = time.time() - start
        total_steps = (step - start_step) * num_envs
        print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
        print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
