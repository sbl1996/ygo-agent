import os
import random
import time
from collections import deque
from queue import Queue
from dataclasses import dataclass, field
from typing import Optional, List


import ygoenv
import optree
import numpy as np
import tyro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics, to_tensor, load_embeddings
from ygoai.rl.agent import PPOAgent as Agent
from ygoai.rl.dist import reduce_gradidents, setup, fprint
from ygoai.rl.buffer import create_obs
from ygoai.rl.ppo import bootstrap_value_selfplay
from ygoai.rl.eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    port: int = 29500
    """the port to use for distributed training"""

    # Algorithm specific arguments
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
    embedding_file: Optional[str] = None
    """the embedding file for card embeddings"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 32
    """the number of history actions to use"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    checkpoint: Optional[str] = None
    """the checkpoint to load the model from"""

    total_timesteps: int = 2000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 128
    """the number of parallel game environments per actor"""
    num_actor_threads: int = 1
    "the number of actor threads to use"
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.98
    """the lambda for the general advantage estimation"""

    num_minibatches: int = 4
    "the number of mini-batches"
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    learn_opponent: bool = True
    """if toggled, the samples from the opponent will be used to train the agent"""
    collect_length: Optional[int] = None
    """the length of the buffer, only the first `num_steps` will be used for training (partial GAE)"""

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"

    compile: Optional[str] = None
    """Compile mode of torch.compile, None for no compilation"""
    local_torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2)"""
    local_env_threads: Optional[int] = 16
    """the number of threads to use for envpool in each actor"""
    fp16_train: bool = False
    """if toggled, training will be done in fp16 precision"""
    fp16_eval: bool = False
    """if toggled, evaluation will be done in fp16 precision"""

    tb_dir: str = "./runs"
    """tensorboard log directory"""
    ckpt_dir: str = "./checkpoints"
    """checkpoint directory"""
    save_interval: int = 500
    """the number of iterations to save the model"""
    log_p: float = 1.0
    """the probability of logging"""
    eval_episodes: int = 128
    """the number of episodes to evaluate the model"""
    eval_interval: int = 50
    """the number of iterations to evaluate the model"""

    # to be filled in runtime
    num_envs: int = 0
    """the number of parallel game environments"""
    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


def make_env(args, num_envs, num_threads, mode='self'):
    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=num_threads,
        seed=args.seed,
        deck1=args.deck1,
        deck2=args.deck2,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        play_mode='self',
    )
    envs.num_envs = num_envs
    envs = RecordEpisodeStatistics(envs)
    return envs


def actor(
    args,
    a_rank,
    rollout_queues: List[Queue],
    param_queue: Queue,
    run_name,
    device_thread_id,
):
    if a_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tb_dir, run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None
    torch.set_num_threads(args.local_torch_threads)
    torch.set_float32_matmul_precision('high')

    device = torch.device(f"cuda:{device_thread_id}" if torch.cuda.is_available() and args.cuda else "cpu")

    deck = init_ygopro(args.env_id, "english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, args.local_num_envs, args.local_env_threads)
    obs_space = envs.env.observation_space
    action_shape = envs.env.action_space.shape
    if a_rank == 0:
        fprint(f"obs_space={obs_space}, action_shape={action_shape}")

    envs_per_thread = args.local_num_envs // args.local_env_threads
    local_eval_episodes = args.eval_episodes // args.world_size
    local_eval_num_envs = local_eval_episodes
    local_eval_num_threads = max(1, local_eval_num_envs // envs_per_thread)
    eval_envs = make_env(args, local_eval_num_envs, local_eval_num_threads, mode='bot')
    
    if args.embedding_file:
        embeddings = load_embeddings(args.embedding_file, args.code_list_file)
        embedding_shape = embeddings.shape
    else:
        embedding_shape = None
    L = args.num_layers
    agent = Agent(args.num_channels, L, L, embedding_shape).to(device)
    agent.eval()

    def predict_step(agent: Agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits, value, valid = agent(next_obs)
        return logits, value

    if args.compile:
        predict_step = torch.compile(predict_step, mode=args.compile)
        agent_r = agent
    else:
        agent_r = agent

    obs = create_obs(obs_space, (args.num_steps, args.local_num_envs), device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.local_num_envs), dtype=torch.bool).to(device)
    values = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    learns = torch.zeros((args.num_steps, args.local_num_envs), dtype=torch.bool).to(device)
    avg_ep_returns = deque(maxlen=1000)
    avg_win_rates = deque(maxlen=1000)

    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = to_tensor(next_obs, device, dtype=torch.uint8)
    next_to_play_ = info["to_play"]
    next_to_play = to_tensor(next_to_play_, device)
    next_done = torch.zeros(args.local_num_envs, device=device, dtype=torch.bool)
    ai_player1_ = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(ai_player1_)
    ai_player1 = to_tensor(ai_player1_, device, dtype=next_to_play.dtype)
    next_value1 = next_value2 = 0
    step = 0
    params_buffer = param_queue.get()[1]

    for iteration in range(1, args.num_iterations):
        if iteration > 2:
            param_queue.get()
        agent.load_state_dict(params_buffer)

        model_time = 0
        env_time = 0
        collect_start = time.time()
        while step < args.num_steps:
            for key in obs:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            learn = next_to_play == ai_player1
            learns[step] = learn

            _start = time.time()
            logits, value = predict_step(agent_r, next_obs)
            value = value.flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

            values[step] = value
            actions[step] = action
            logprobs[step] = logprob
            action = action.cpu().numpy()
            model_time += time.time() - _start

            next_nonterminal = 1 - next_done.float()
            next_value1 = torch.where(learn, value, next_value1) * next_nonterminal
            next_value2 = torch.where(learn, next_value2, value) * next_nonterminal

            _start = time.time()
            to_play = next_to_play_
            next_obs, reward, next_done_, info = envs.step(action)
            next_to_play_ = info["to_play"]
            next_to_play = to_tensor(next_to_play_, device)
            env_time += time.time() - _start
            rewards[step] = to_tensor(reward, device)
            next_obs, next_done = to_tensor(next_obs, device, torch.uint8), to_tensor(next_done_, device, torch.bool)
            step += 1

            global_step += args.num_envs

            if not writer:
                continue

            for idx, d in enumerate(next_done_):
                if d:
                    pl = 1 if to_play[idx] == ai_player1_[idx] else -1
                    episode_length = info['l'][idx]
                    episode_reward = info['r'][idx] * pl
                    win = 1 if episode_reward > 0 else 0
                    avg_ep_returns.append(episode_reward)
                    avg_win_rates.append(win)

                    if random.random() < args.log_p:
                        n = 100
                        if random.random() < 10/n or iteration <= 1:
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            fprint(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if random.random() < 1/n:
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_ep_returns), global_step)
                            writer.add_scalar("charts/avg_win_rate", np.mean(avg_win_rates), global_step)

        collect_time = time.time() - collect_start
        fprint(f"collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        _start = time.time()
        # bootstrap value if not done
        with torch.no_grad():
            value = predict_step(agent_r, next_obs)[1].reshape(-1)
        nextvalues1 = torch.where(next_to_play == ai_player1, value, next_value1)
        nextvalues2 = torch.where(next_to_play != ai_player1, value, next_value2)

        step = 0

        for iq, rq in enumerate(rollout_queues):
            n_e = args.local_num_envs // len(rollout_queues)
            start = iq * n_e
            end = start + n_e
            data = []
            d = optree.tree_map(lambda x: x[:, start:end],
                (obs, actions, logprobs, rewards, dones, values, learns))
            for v in d:
                data.append(v)
            for v in [next_done, nextvalues1, nextvalues2]:
                data.append(v[start:end])
            rq.put(data)


        SPS = int((global_step - warmup_steps) / (time.time() - start_time))

        # Warmup at first few iterations for accurate SPS measurement
        SPS_warmup_iters = 10
        if iteration == SPS_warmup_iters:
            start_time = time.time()
            warmup_steps = global_step
        if iteration > SPS_warmup_iters:
            if a_rank == 0:
                fprint(f"SPS: {SPS}")

        if args.eval_interval and iteration % args.eval_interval == 0:
            # Eval with rule-based policy
            _start = time.time()
            eval_return = evaluate(
                eval_envs, agent_r, local_eval_episodes, device, args.fp16_eval)[0]
            eval_stats = torch.tensor(eval_return, dtype=torch.float32, device=device)

            # sync the statistics
            # if args.world_size > 1:
            #     dist.all_reduce(eval_stats, op=dist.ReduceOp.AVG)
            eval_return = eval_stats.cpu().numpy()
            if a_rank == 0:
                writer.add_scalar("charts/eval_return", eval_return, global_step)
                eval_time = time.time() - _start
                fprint(f"eval_time={eval_time:.4f}, eval_ep_return={eval_return:.4f}")


def learner(
    args: Args,
    l_rank,
    rollout_queue: Queue,
    param_queue: Queue,
    run_name,
    ckpt_dir,
    device_thread_id,    
):
    num_learners = len(args.learner_device_ids)
    if len(args.learner_device_ids) > 1:
        setup('nccl', l_rank, num_learners, args.port)
    local_batch_size = args.local_batch_size // num_learners
    local_minibatch_size = args.local_minibatch_size // num_learners

    torch.set_num_threads(args.local_torch_threads)
    torch.set_float32_matmul_precision('high')

    args.seed += l_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - l_rank)

    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{device_thread_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.embedding_file:
        embeddings = load_embeddings(args.embedding_file, args.code_list_file)
        embedding_shape = embeddings.shape
    else:
        embedding_shape = None
    L = args.num_layers
    agent = Agent(args.num_channels, L, L, embedding_shape).to(device)

    from ygoai.rl.ppo import train_step
    if args.compile:
        train_step = torch.compile(train_step, mode=args.compile)

    optim_params = list(agent.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    global_step = 0

    first_in_group = l_rank % (num_learners // (len(args.actor_device_ids) * args.num_actor_threads)) == 0

    if first_in_group:
        param_queue.put(("Init", agent.state_dict()))

    for iteration in range(1, args.num_iterations):
        bootstrap_start = time.time()
        _start = time.time()
        data = rollout_queue.get()
        wait_time = time.time() - _start
        obs, actions, logprobs, rewards, dones, values, learns, next_done, nextvalues1, nextvalues2 \
            = optree.tree_map(lambda x: x.to(device=device, non_blocking=True), data)
        advantages = bootstrap_value_selfplay(
            values, rewards, dones, learns, nextvalues1, nextvalues2, next_done, args.gamma, args.gae_lambda)
        bootstrap_time = time.time() - bootstrap_start

        _start = time.time()
        # flatten the batch
        b_obs = {
            k: v[:args.num_steps].reshape((-1,) + v.shape[2:])
            for k, v in obs.items()
        }
        b_actions = actions[:args.num_steps].flatten(0, 1)
        b_logprobs = logprobs[:args.num_steps].reshape(-1)
        b_advantages = advantages[:args.num_steps].reshape(-1)
        b_values = values[:args.num_steps].reshape(-1)
        b_returns = b_advantages + b_values
        if args.learn_opponent:
            b_learns = torch.ones_like(b_values, dtype=torch.bool)
        else:
            b_learns = learns[:args.num_steps].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, local_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {
                    k: v[mb_inds] for k, v in b_obs.items()
                }
                old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                    train_step(agent, optimizer, scaler, mb_obs, b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds], b_learns[mb_inds], args)
                reduce_gradidents(optim_params, num_learners)
                nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

        global_step += args.num_envs

        if first_in_group:
            param_queue.put(("Done", None))

        if l_rank == 0:
            train_time = time.time() - _start
            fprint(f"train_time={train_time:.4f}, bootstrap_time={bootstrap_time:.4f}, wait_time={wait_time:.4f}")

            if iteration % args.save_interval == 0:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent.pt"))

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            fprint(f"global_step={global_step}, value_loss={v_loss.item():.4f}, policy_loss={pg_loss.item():.4f}, entropy_loss={entropy_loss.item():.4f}")

            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = tyro.cli(Args)
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"

    args.world_size = 1
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_iterations = args.total_timesteps // args.batch_size
    args.env_threads = args.local_env_threads * args.num_actor_threads * len(args.actor_device_ids)
    args.local_torch_threads = args.local_torch_threads or int(os.getenv("OMP_NUM_THREADS", "2"))

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    rollout_queues = []
    param_queues = []

    actor_processes = []
    learner_processes = []

    num_actors = len(args.actor_device_ids) * args.num_actor_threads
    num_learners = len(args.learner_device_ids)
    assert num_learners % num_actors == 0, "num_learners must be divisible by num_actors"
    group_size = num_learners // num_actors

    for i, device_id in enumerate(args.actor_device_ids):
        for j in range(args.num_actor_threads):
            a_rank = i * args.num_actor_threads + j
            param_queues.append(mp.Queue(maxsize=1))
            rollout_queues_ = [mp.Queue(maxsize=1) for _ in range(group_size)]
            rollout_queues.extend(rollout_queues_)
            p = mp.Process(
                target=actor,
                args=(args, a_rank, rollout_queues_, param_queues[-1], run_name, device_id),
            )
            actor_processes.append(p)
            p.start()

    for i, device_id in enumerate(args.learner_device_ids):
        param_queue = param_queues[i // group_size]
        rollout_queue = rollout_queues[i]
        p = mp.Process(
            target=learner,
            args=(args, i, rollout_queue, param_queue, run_name, ckpt_dir, device_id),
        )
        learner_processes.append(p)
        p.start()

    for p in actor_processes + learner_processes:
        p.join()