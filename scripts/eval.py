import sys
import time
import os
import random
from typing import Optional, Literal
from dataclasses import dataclass

import ygoenv
import numpy as np

import optree

import tyro

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.agent import PPOAgent as Agent
from ygoai.rl.buffer import create_obs


@dataclass
class Args:
    seed: int = 1
    """the random seed"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    env_id: str = "YGOPro-v0"
    """the id of the environment"""
    deck: str = "../assets/deck/OldSchool.ydk"
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
    n_history_actions: int = 16
    """the number of history actions to use"""

    player: int = -1
    """the player to play as, -1 means random, 0 is the first player, 1 is the second player"""
    play: bool = False
    """whether to play the game"""
    selfplay: bool = False
    """whether to use selfplay"""
    record: bool = False
    """whether to record the game as YGOPro replays"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""
    verbose: bool = False
    """whether to print debug information"""

    bot_type: Literal["random", "greedy"] = "greedy"
    """the type of bot to use"""
    strategy: Literal["random", "greedy"] = "greedy"
    """the strategy to use if agent is not used"""

    agent: bool = False
    """whether to use the agent"""
    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    checkpoint: Optional[str] = "checkpoints/agent.pt"
    """the checkpoint to load"""

    compile: bool = False
    """if toggled, the model will be compiled"""
    optimize: bool = True
    """if toggled, the model will be optimized"""
    torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2) * world_size"""
    env_threads: Optional[int] = 16
    """the number of threads to use for envpool, defaults to `num_envs`"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.play:
        args.num_envs = 1
        args.verbose = True
    
    if args.record:
        assert args.num_envs == 1, "Recording only works with a single environment"
        assert args.verbose, "Recording only works with verbose mode"

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)
    args.torch_threads = args.torch_threads or int(os.getenv("OMP_NUM_THREADS", "4"))

    deck = init_ygopro(args.lang, args.deck, args.code_list_file)

    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    if args.agent:
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        torch.set_num_threads(args.torch_threads)
        torch.set_float32_matmul_precision('high')

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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
        play_mode='human' if args.play else ('self' if args.selfplay else ('bot' if args.bot_type == "greedy" else "random")),
        verbose=args.verbose,
        record=args.record,
    )
    envs.num_envs = num_envs
    envs = RecordEpisodeStatistics(envs)

    if args.agent:
        # count lines of code_list
        with open(args.code_list_file, "r") as f:
            code_list = f.readlines()
            embedding_shape = len(code_list)
        L = args.num_layers
        agent = Agent(args.num_channels, L, L, 1, embedding_shape).to(device)
        agent = agent.eval()
        if args.checkpoint:
            state_dict = torch.load(args.checkpoint, map_location=device)
        else:
            state_dict = None

        if args.compile:
            agent = torch.compile(agent, mode='reduce-overhead')
            if state_dict:
                agent.load_state_dict(state_dict)
        else:
            prefix = "_orig_mod."
            if state_dict:
                state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
                agent.load_state_dict(state_dict)
            
            if args.optimize:
                obs = create_obs(envs.observation_space, (num_envs,), device=device)
                with torch.no_grad():
                    traced_model = torch.jit.trace(agent, (obs,), check_tolerance=False, check_trace=False)
                    agent = torch.jit.optimize_for_inference(traced_model)

    obs, infos = envs.reset()

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    win_reasons = []

    step = 0
    start = time.time()
    start_step = step

    model_time = env_time = 0
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        if args.agent:
            _start = time.time()
            obs = optree.tree_map(lambda x: torch.from_numpy(x).to(device=device), obs)
            with torch.no_grad():
                logits, values = agent(obs)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()
            if args.play:
                print(probs[probs != 0].tolist())
                print(values)
            actions = probs.argmax(axis=1)
            model_time += time.time() - _start
        else:
            if args.strategy == "random":
                actions = np.random.randint(infos['num_options'])
            else:
                actions = np.zeros(num_envs, dtype=np.int32)

        # for k, v in obs.items():
        #     v = v[0]
        #     if k == 'cards_':
        #         v = np.concatenate([np.arange(v.shape[0])[:, None], v], axis=1)
        #     print(k, v.tolist())
        # print(infos)
        # print(actions[0])

        _start = time.time()
        obs, rewards, dones, infos = envs.step(actions)
        env_time += time.time() - _start

        step += 1

        for idx, d in enumerate(dones):
            if d:
                win_reason = infos['win_reason'][idx]
                episode_length = infos['l'][idx]
                episode_reward = infos['r'][idx]
                if args.selfplay:
                    pl = 1 if infos['to_play'][idx] == 0 else -1
                    winner = 0 if episode_reward * pl > 0 else 1
                    win = 1 - winner
                else:
                    if episode_reward < 0:
                        win = 0
                    else:
                        win = 1

                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                win_rates.append(win)
                win_reasons.append(1 if win_reason == 1 else 0)
                sys.stderr.write(f"Episode {len(episode_lengths)}: length={episode_length}, reward={episode_reward}, win={win}, win_reason={win_reason}\n")
        if len(episode_lengths) >= args.num_episodes:
            break

    print(f"len={np.mean(episode_lengths)}, reward={np.mean(episode_rewards)}, win_rate={np.mean(win_rates)}, win_reason={np.mean(win_reasons)}")
    if not args.play:
        total_time = time.time() - start
        total_steps = (step - start_step) * num_envs
        print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
        print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
    