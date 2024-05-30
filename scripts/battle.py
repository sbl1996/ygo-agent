import sys
import time
import os
import random
from typing import Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from functools import partial

import ygoenv
import numpy as np

import tyro

import jax
import jax.numpy as jnp
import flax

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.jax.agent import RNNAgent, ModelArgs


@dataclass
class Args:
    seed: int = 1
    """the random seed"""

    env_id1: str = "YGOPro-v1"
    """the id of the environment1"""
    env_id2: Optional[str] = None
    """the id of the environment2, defaults to `env_id1`"""
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
    n_history_actions1: int = 32
    """the number of history actions to use for the environment1"""
    n_history_actions2: Optional[int] = None
    """the number of history actions to use for the environment2, defaults to `n_history_actions1`"""
    num_embeddings: Optional[int] = None
    """the number of embeddings of the agent"""
    accurate: bool = True
    """whether to do accurate evaluation. If not, there will be more short games"""
    reverse: bool = False
    """whether to reverse the order of the agents"""

    verbose: bool = False
    """whether to print debug information"""
    record: bool = False
    """whether to record the game as YGOPro replays"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""

    m1: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent1"""
    m2: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent2"""

    checkpoint1: str = "checkpoints/agent.pt"
    """the checkpoint to load for the first agent, must be a `flax_model` file"""
    checkpoint2: str = "checkpoints/agent.pt"
    """the checkpoint to load for the second agent, must be a `flax_model` file"""
    
    xla_device: Optional[str] = None
    """the XLA device to use, `cpu` for forcing running on CPU"""

    env_threads: Optional[int] = None
    """the number of threads to use for envpool, defaults to `num_envs`"""


def create_agent1(args):
    return RNNAgent(
        **asdict(args.m1),
        embedding_shape=args.num_embeddings,
    )


def create_agent2(args):
    return RNNAgent(
        **asdict(args.m2),
        embedding_shape=args.num_embeddings,
    )


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

    args = tyro.cli(Args)

    if args.record:
        args.num_envs = 1
        args.verbose = True
        print("Set num_envs=1 and verbose=True for recording")
        if not os.path.exists("replay"):
            os.makedirs("replay")

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)

    if args.env_id2 is None:
        args.env_id2 = args.env_id1
    if args.n_history_actions2 is None:
        args.n_history_actions2 = args.n_history_actions1

    cross_env = args.env_id1 != args.env_id2
    env_id1 = args.env_id1
    env_id2 = args.env_id2

    deck1 = init_ygopro(env_id1, args.lang, args.deck, args.code_list_file)
    if not cross_env:
        deck2 = deck1
    else:
        deck2 = init_ygopro(env_id2, args.lang, args.deck, args.code_list_file)

    args.deck1 = args.deck1 or deck1
    args.deck2 = args.deck2 or deck2

    seed = args.seed + 100000
    random.seed(seed)
    seed = random.randint(0, 1e8)
    random.seed(seed)
    np.random.seed(seed)

    if args.xla_device is not None:
        os.environ.setdefault("JAX_PLATFORMS", args.xla_device)

    if args.accurate:
        if args.num_envs != args.num_episodes:
            args.num_envs = args.num_episodes
            print("Set num_envs to num_episodes for accurate evaluation")

    num_envs = args.num_envs

    env_option = dict(
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=args.env_threads,
        seed=seed,
        player=-1,
        max_options=args.max_options,
        play_mode='self',
        async_reset=False,
        verbose=args.verbose,
        record=args.record,        
    )
    envs1 = ygoenv.make(
        task_id=env_id1,
        n_history_actions=args.n_history_actions1,
        deck1=args.deck1,
        deck2=args.deck2,
        **env_option,
    )
    if cross_env:
        envs2 = ygoenv.make(
            task_id=env_id2,
            n_history_actions=args.n_history_actions2,
            deck1=deck2,
            deck2=deck2,
            **env_option,
        )

    key = jax.random.PRNGKey(seed)

    obs_space1 = envs1.observation_space
    envs1.num_envs = num_envs
    envs1 = RecordEpisodeStatistics(envs1)
    sample_obs1 = jax.tree.map(lambda x: jnp.array([x]), obs_space1.sample())
    agent1 = create_agent1(args)
    rstate1 = agent1.init_rnn_state(1)
    params1 = jax.jit(agent1.init)(key, sample_obs1, rstate1)
    with open(args.checkpoint1, "rb") as f:
        params1 = flax.serialization.from_bytes(params1, f.read())

    if cross_env:
        obs_space2 = envs2.observation_space
        envs2.num_envs = num_envs
        envs2 = RecordEpisodeStatistics(envs2)
        sample_obs2 = jax.tree.map(lambda x: jnp.array([x]), obs_space2.sample())
    else:
        sample_obs2 = sample_obs1

    if args.checkpoint1 == args.checkpoint2:
        params2 = params1
    else:
        agent2 = create_agent2(args)
        rstate2 = agent2.init_rnn_state(1)
        params2 = jax.jit(agent2.init)(key, sample_obs2, rstate2)
        with open(args.checkpoint2, "rb") as f:
            params2 = flax.serialization.from_bytes(params2, f.read())
    
    params1 = jax.device_put(params1)
    params2 = jax.device_put(params2)
    
    @partial(jax.jit, static_argnums=(4,))
    def get_probs(params, rstate, obs, done=None, model_id=1):
        if model_id == 1:
            agent = create_agent1(args)
        else:
            agent = create_agent2(args)
        next_rstate, logits = agent.apply(params, obs, rstate)[:2]
        probs = jax.nn.softmax(logits, axis=-1)
        if done is not None:
            next_rstate = jnp.where(done[:, None], 0, next_rstate)
        return next_rstate, probs

    if num_envs != 1:
        @jax.jit
        def get_probs2(params1, params2, rstate1, rstate2, obs1, obs2, main, done):
            next_rstate1, probs1 = get_probs(params1, rstate1, obs1, None, 1)
            next_rstate2, probs2 = get_probs(params2, rstate2, obs2, None, 2)
            probs = jnp.where(main[:, None], probs1, probs2)
            rstate1 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x1, x2), next_rstate1, rstate1)
            rstate2 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x2, x1), next_rstate2, rstate2)
            rstate1, rstate2 = jax.tree.map(
                lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
            return rstate1, rstate2, probs

        def predict_fn(rstate1, rstate2, obs1, obs2, main, done):
            rstate1, rstate2, probs = get_probs2(params1, params2, rstate1, rstate2, obs1, obs2, main, done)
            return rstate1, rstate2, np.array(probs)
    else:
        def predict_fn(rstate1, rstate2, obs1, obs2, main, done):
            if main[0]:
                rstate1, probs = get_probs(params1, rstate1, obs1, done, 1)
            else:
                rstate2, probs = get_probs(params2, rstate2, obs2, done, 2)
            return rstate1, rstate2, np.array(probs)

    obs1, infos1 = envs1.reset()
    next_to_play1 = infos1['to_play']
    if cross_env:
        obs2, infos2 = envs2.reset()
        next_to_play2 = infos2['to_play']
        np.testing.assert_array_equal(next_to_play1, next_to_play2)
    else:
        obs2 = obs1

    dones = np.zeros(num_envs, dtype=np.bool_)

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    win_reasons = []
    win_players = []
    win_agents = []

    step = 0
    start = time.time()
    start_step = step

    first_player = np.zeros(num_envs // 2, dtype=np.int64)
    second_player = np.ones(num_envs - num_envs // 2, dtype=np.int64)
    if args.reverse:
        main_player = np.concatenate([second_player, first_player])
    else:
        main_player = np.concatenate([first_player, second_player])
    # main_player = np.zeros(num_envs, dtype=np.int64)
    # main_player = np.ones(num_envs, dtype=np.int64)
    rstate1 = agent1.init_rnn_state(num_envs)
    rstate2 = agent2.init_rnn_state(num_envs)
    collected = np.zeros((args.num_episodes,), dtype=np.bool_)

    if not args.verbose:
        pbar = tqdm(total=args.num_episodes)

    model_time = env_time = 0
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        _start = time.time()
        main = next_to_play1 == main_player
        rstate1, rstate2, probs = predict_fn(rstate1, rstate2, obs1, obs2, main, dones)

        actions = probs.argmax(axis=1)
        model_time += time.time() - _start

        to_play1 = next_to_play1

        _start = time.time()
        obs1, rewards1, dones1, infos1 = envs1.step(actions)
        next_to_play1 = infos1['to_play']
        if cross_env:
            obs2, rewards2, dones2, infos2 = envs2.step(actions)
            next_to_play2 = infos2['to_play']
            np.testing.assert_array_equal(next_to_play1, next_to_play2)
            np.testing.assert_array_equal(dones1, dones2)
        else:
            obs2 = obs1

        env_time += time.time() - _start

        step += 1

        for idx, d in enumerate(dones1):
            if not d or (args.accurate and collected[idx]):
                continue
            collected[idx] = True
            win_reason = infos1['win_reason'][idx]
            pl = 1 if main[idx] else -1
            episode_length = infos1['l'][idx]
            episode_reward = infos1['r'][idx]
            main_reward = episode_reward * pl
            win = int(main_reward > 0)

            win_player = 0 if (to_play1[idx] == 0 and episode_reward > 0) or (to_play1[idx] == 1 and episode_reward < 0) else 1
            win_players.append(win_player)
            win_agent = 1 if main_reward > 0 else 2
            win_agents.append(win_agent)
            # print(f"{len(episode_lengths)}: {episode_length}, {main_reward}")
            episode_lengths.append(episode_length)
            episode_rewards.append(main_reward)
            win_rates.append(win)
            win_reasons.append(1 if win_reason == 1 else 0)
            if args.verbose:
                sys.stderr.write(f"Episode {len(episode_lengths)}: length={episode_length}, reward={main_reward}, win={win}, win_reason={win_reason}\n")
            else:
                pbar.set_postfix(len=np.mean(episode_lengths), reward=np.mean(episode_rewards), win_rate=np.mean(win_rates))
                pbar.update(1)

            # Only when num_envs=1, we switch the player here
            if args.verbose:
                main_player = 1 - main_player
            else:
                main_player[idx] = 1 - main_player[idx]

        if len(episode_lengths) >= args.num_episodes:
            break

    if not args.verbose:
        pbar.close()
    print(f"len={np.mean(episode_lengths)}, reward={np.mean(episode_rewards)}, win_rate={np.mean(win_rates)}, win_reason={np.mean(win_reasons)}")

    episode_lengths = np.array(episode_lengths)
    win_players = np.array(win_players)
    win_agents = np.array(win_agents)
    N = len(win_players)

    mask1 = (win_players == 0) & (win_agents == 1)
    mask2 = (win_players == 0) & (win_agents == 2)
    mask3 = (win_players == 1) & (win_agents == 1)
    mask4 = (win_players == 1) & (win_agents == 2)

    N1, N2, N3, N4 = [np.sum(m) for m in [mask1, mask2, mask3, mask4]]

    print(f"Payoff matrix:")
    w1 = N1 / N
    w2 = N2 / N
    w3 = N3 / N
    w4 = N4 / N
    print(f"   agent1  agent2")
    print(f"0  {w1:.4f}  {w2:.4f}")
    print(f"1  {w3:.4f}  {w4:.4f}")

    print(f"0/1 matrix, win rates of agentX as playerY")
    w1 = N1 / (N1 + N4)
    w2 = N2 / (N2 + N3)
    w3 = 1 - w2
    w4 = 1 - w1
    print(f"   agent1  agent2")
    print(f"0  {w1:.4f}  {w2:.4f}")
    print(f"1  {w3:.4f}  {w4:.4f}")

    print(f"Length matrix, length of games of agentX as playerY")
    l1 = np.mean(episode_lengths[mask1 | mask4])
    l2 = np.mean(episode_lengths[mask2 | mask3])
    print(f"   agent1  agent2")
    print(f"0  {l1:3.2f}  {l2:3.2f}")
    print(f"1  {l2:3.2f}  {l1:3.2f}")

    total_time = time.time() - start
    total_steps = (step - start_step) * num_envs
    print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
    print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
