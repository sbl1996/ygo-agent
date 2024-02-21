import os
import random
import time
from typing import Optional, Literal
from dataclasses import dataclass

import ygoenv
import numpy as np
import optree
import tyro

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics, Elo
from ygoai.rl.agent import Agent
from ygoai.rl.buffer import DMCDictBuffer
from ygoai.rl.dist import reduce_gradidents


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
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
    embedding_file: str = "embeddings_en.npy"
    """the embedding file for card embeddings"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 8
    """the number of history actions to use"""
    play_mode: str = "self"
    """the play mode, can be combination of 'self', 'bot', 'random', like 'self+bot'"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""

    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps per env per iteration"""
    buffer_size: int = 200000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    minibatch_size: int = 256
    """the mini-batch size"""
    eps: float = 0.05
    """the epsilon for exploration"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    log_p: float = 0.1
    """the probability of logging"""
    save_freq: int = 100
    """the saving frequency (in terms of iterations)"""

    backend: Literal["gloo", "nccl", "mpi"] = "nccl"
    """the backend for distributed training"""
    compile: bool = True
    """if toggled, model will be compiled for better performance"""
    torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2) * world_size"""
    env_threads: Optional[int] = 32
    """the number of threads to use for envpool, defaults to `num_envs`"""

    tb_dir: str = "./runs"
    """tensorboard log directory"""
    port: int = 12355
    """the port to use for distributed training"""

    # to be filled in runtime
    local_buffer_size: int = 0
    """the local buffer size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    local_num_envs: int = 0
    """the number of parallel game environments (in the local rank, computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


def setup(backend, rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def run(local_rank, world_size):
    args = tyro.cli(Args)
    args.world_size = world_size
    args.local_num_envs = args.num_envs // args.world_size
    args.local_minibatch_size = int(args.minibatch_size // args.world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.local_buffer_size = int(args.buffer_size // args.world_size)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.env_threads = args.env_threads or args.num_envs
    args.torch_threads = args.torch_threads or (int(os.getenv("OMP_NUM_THREADS", "2")) * args.world_size)

    local_torch_threads = args.torch_threads // args.world_size
    local_env_threads = args.env_threads // args.world_size

    torch.set_num_threads(local_torch_threads)
    torch.set_float32_matmul_precision('high')

    if args.world_size > 1:
        setup(args.backend, local_rank, args.world_size, args.port)

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    writer = None
    if local_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tb_dir, run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )


    # TRY NOT TO MODIFY: seeding
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    deck = init_ygopro("english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=args.local_num_envs,
        num_threads=local_env_threads,
        seed=args.seed,
        deck1=args.deck1,
        deck2=args.deck2,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        play_mode=args.play_mode,
    )
    envs.num_envs = args.local_num_envs
    obs_space = envs.observation_space
    action_space = envs.action_space
    if local_rank == 0:
        print(f"obs_space={obs_space}, action_space={action_space}")

    envs = RecordEpisodeStatistics(envs)

    embeddings = np.load(args.embedding_file)
    L = args.num_layers
    agent = Agent(args.num_channels, L, L, 1, embeddings.shape).to(device)
    agent.load_embeddings(embeddings)

    if args.compile:
        agent = torch.compile(agent, mode='reduce-overhead')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    avg_win_rates = []
    avg_ep_returns = []
    # elo = Elo()

    selfplay = "self" in args.play_mode
    rb = DMCDictBuffer(
        args.local_buffer_size,
        obs_space,
        action_space,
        device=device,
        n_envs=args.local_num_envs,
        selfplay=selfplay,
    )

    gamma = np.float32(args.gamma)

    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    obs, infos = envs.reset()
    num_options = infos['num_options']
    to_play = infos['to_play'] if selfplay else None
    for iteration in range(1, args.num_iterations + 1):
        agent.eval()
        model_time = 0
        env_time = 0
        buffer_time = 0

        collect_start = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs

            obs = optree.tree_map(lambda x: torch.from_numpy(x).to(device=device), obs)
            if random.random() < args.eps:
                actions_ = np.random.randint(num_options)
                actions = torch.from_numpy(actions_).to(device)
            else:
                _start = time.time()
                with torch.no_grad():
                    values = agent(obs)[0]
                actions = torch.argmax(values, dim=1)
                actions_ = actions.cpu().numpy()
                model_time += time.time() - _start

            _start = time.time()
            next_obs, rewards, dones, infos = envs.step(actions_)
            env_time += time.time() - _start
            num_options = infos['num_options']

            _start = time.time()
            rb.add(obs, actions, rewards, to_play)
            buffer_time += time.time() - _start

            obs = next_obs
            to_play = infos['to_play'] if selfplay else None

            for idx, d in enumerate(dones):
                if d:
                    _start = time.time()
                    rb.mark_episode(idx, gamma)
                    buffer_time += time.time() - _start

                    if writer and random.random() < args.log_p:
                        episode_length = infos['l'][idx]
                        episode_reward = infos['r'][idx]
                        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        if selfplay:
                            if infos['is_selfplay'][idx]:
                                # win rate for the first player
                                pl = 1 if to_play[idx] == 0 else -1
                                winner = 0 if episode_reward * pl > 0 else 1
                                avg_win_rates.append(1 - winner)
                            else:
                                # win rate of agent
                                winner = 0 if episode_reward > 0 else 1
                                # elo.update(winner)
                        else:
                            avg_ep_returns.append(episode_reward)
                            winner = 0 if episode_reward > 0 else 1
                            avg_win_rates.append(1 - winner)
                            # elo.update(winner)
                        print(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if len(avg_win_rates) > 100:
                            writer.add_scalar("charts/avg_win_rate", np.mean(avg_win_rates), global_step)
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_ep_returns), global_step)
                            avg_win_rates = []
                            avg_ep_returns = []

        collect_time = time.time() - collect_start
        if writer:
            print(f"global_step={global_step}, collect_time={collect_time}, model_time={model_time}, env_time={env_time}, buffer_time={buffer_time}")

        agent.train()
        train_start = time.time()
        model_time = 0
        sample_time = 0

        # ALGO LOGIC: training.
        _start = time.time()

        if not rb.full:
            continue
        b_inds = rb.get_data_indices()
        np.random.shuffle(b_inds)
        b_obs, b_actions, b_returns = rb._get_samples(b_inds)
        print(f"{len(b_inds)}, {b_returns.shape}, {args.local_buffer_size}, {args.local_minibatch_size}")

        n_samples = torch.tensor(b_returns.shape[0], device=device, dtype=torch.int64)
        dist.all_reduce(n_samples, op=dist.ReduceOp.MIN)
        n_samples = n_samples.item()
        print(f"n_samples={n_samples}")
        raise ValueError
        sample_time += time.time() - _start
        for start in range(0, len(b_returns), args.local_minibatch_size):
            _start = time.time()
            end = start + args.local_minibatch_size
            mb_obs = {
                k: v[start:end] for k, v in b_obs.items()
            }
            mb_actions = b_actions[start:end]
            mb_returns = b_returns[start:end]
            sample_time += time.time() - _start

            _start = time.time()
            outputs, valid = agent(mb_obs)
            outputs = torch.gather(outputs, 1, mb_actions).squeeze(1)
            outputs = torch.where(valid, outputs, mb_returns)
            loss = F.mse_loss(mb_returns, outputs)
            loss = loss * (args.local_minibatch_size / valid.float().sum())

            optimizer.zero_grad()
            loss.backward()
            reduce_gradidents(agent)
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            model_time += time.time() - _start

        if not rb.full or iteration % 10 == 0:
            torch.cuda.empty_cache()

        train_time = time.time() - train_start

        if writer:
            print(f"global_step={global_step}, train_time={train_time}, model_time={model_time}, sample_time={sample_time}")

            writer.add_scalar("losses/value_loss", loss.item(), global_step)
            writer.add_scalar("losses/q_values", outputs.mean().item(), global_step)

            if iteration == 10:
                warmup_steps = global_step
                start_time = time.time()
            if iteration > 10:
                SPS = int((global_step - warmup_steps) / (time.time() - start_time))
                print("SPS:", SPS)
                writer.add_scalar("charts/SPS", SPS, global_step)

            if iteration % args.save_freq == 0:
                save_path = f"checkpoints/agent.pt"
                print(f"Saving model to {save_path}")
                torch.save(agent.state_dict(), save_path)

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        run(local_rank=0, world_size=world_size)
    else:
        children = []
        for i in range(world_size):
            subproc = mp.Process(target=run, args=(i, world_size))
            children.append(subproc)
            subproc.start()

        for i in range(world_size):
            children[i].join()
