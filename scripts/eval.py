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

    player: int = -1
    """the player to play as, -1 means random, 0 is the first player, 1 is the second player"""
    play: bool = False
    """whether to play the game"""
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

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    checkpoint: Optional[str] = None
    """the checkpoint to load, `pt` or `flax_model` file"""

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
    optimize: bool = True
    """if toggled, the model will be optimized"""
    convert: bool = False
    """if toggled, the model will be converted to a jit model and the program will exit"""
    torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2) * world_size"""

    env_threads: Optional[int] = 16
    """the number of threads to use for envpool, defaults to `num_envs`"""

    framework: Optional[Literal["torch", "jax"]] = None


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.play:
        args.num_envs = 1
        args.verbose = True
    
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

    if args.checkpoint and args.framework is None:
        args.framework = "jax" if "flax_model" in args.checkpoint else "torch"

    if args.framework == "torch":
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        args.torch_threads = args.torch_threads or int(os.getenv("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(args.torch_threads)
        torch.set_float32_matmul_precision('high')
    else:
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

    if args.framework == 'torch':
        from ygoai.rl.agent import PPOAgent as Agent
        from ygoai.rl.buffer import create_obs

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.checkpoint.endswith(".ptj"):
            agent = torch.jit.load(args.checkpoint)
        else:
            # count lines of code_list
            embedding_shape = args.num_embeddings
            if embedding_shape is None:
                with open(args.code_list_file, "r") as f:
                    code_list = f.readlines()
                    embedding_shape = len(code_list)
            L = args.num_layers
            agent = Agent(args.num_channels, L, L, embedding_shape).to(device)
            state_dict = torch.load(args.checkpoint, map_location=device)
            if not args.compile:
                prefix = "_orig_mod."
                state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            print(agent.load_state_dict(state_dict))

            if args.compile:
                if args.convert:
                    # Don't support dynamic shapes and very slow inference
                    raise NotImplementedError
                    # obs = create_obs(envs.observation_space, (num_envs,), device=device)
                    # dynamic_shapes = {"x": {}}
                    # # batch_dim = torch.export.Dim("batch", min=1, max=64)
                    # batch_dim = None
                    # for k, v in obs.items():
                    #     dynamic_shapes["x"][k] = {0: batch_dim}
                    # program = torch.export.export(
                    #     agent, (obs,),
                    #     dynamic_shapes=dynamic_shapes,
                    # )
                    # torch.export.save(program, args.checkpoint + "2")
                    # exit(0)
                agent = torch.compile(agent, mode='reduce-overhead')
            elif args.optimize:
                obs = create_obs(envs.observation_space, (num_envs,), device=device)
                def optimize_for_inference(agent):
                    with torch.no_grad():
                        traced_model = torch.jit.trace(agent, (obs,), check_tolerance=False, check_trace=False)
                        return torch.jit.optimize_for_inference(traced_model)
                agent = optimize_for_inference(agent)
                if args.convert:
                    torch.jit.save(agent, args.checkpoint + "j")
                    print(f"Optimized model saved to {args.checkpoint}j")
                    exit(0)

        def predict_fn(obs):
            obs = optree.tree_map(lambda x: torch.from_numpy(x).to(device=device), obs)
            with torch.no_grad():
                logits = agent(obs)[0]
            probs = torch.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()
            return probs
    else:
        import jax
        import jax.numpy as jnp
        import flax
        from ygoai.rl.jax.agent2 import PPOAgent
        def create_agent(args):
            return PPOAgent(
                channels=128,
                num_layers=2,
                embedding_shape=args.num_embeddings,
            )
        agent = create_agent(args)

        key = jax.random.PRNGKey(args.seed)
        key, agent_key = jax.random.split(key, 2)
        sample_obs = jax.tree_map(lambda x: jnp.array([x]), obs_space.sample())
        params = agent.init(agent_key, sample_obs)
        with open(args.checkpoint, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())

        @jax.jit
        def get_probs(
            params: flax.core.FrozenDict,
            next_obs,
        ):
            logits = create_agent(args).apply(params, next_obs)[0]
            return jax.nn.softmax(logits, axis=-1)

        def predict_fn(obs):
            probs = get_probs(params, obs)
            return np.array(probs)

        print(f"loaded checkpoint from {args.checkpoint}")


    obs, infos = envs.reset()
    next_to_play = infos['to_play']

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

        if args.framework:
            _start = time.time()
            probs = predict_fn(obs)
            if args.verbose:
                print([f"{p:.4f}" for p in probs[probs != 0].tolist()])
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
            if d:
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

    print(f"len={np.mean(episode_lengths)}, reward={np.mean(episode_rewards)}, win_rate={np.mean(win_rates)}, win_reason={np.mean(win_reasons)}")
    if not args.play:
        total_time = time.time() - start
        total_steps = (step - start_step) * num_envs
        print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
        print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
    