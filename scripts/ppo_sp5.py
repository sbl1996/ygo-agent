import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional


import ygoenv
import numpy as np
import optree
import tyro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.agent import PPOAgent as Agent
from ygoai.rl.dist import reduce_gradidents, mp_start, setup
from ygoai.rl.buffer import create_obs


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
    embedding_file: Optional[str] = "embeddings_en.npy"
    """the embedding file for card embeddings"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 16
    """the number of history actions to use"""
    play_mode: str = "bot"
    """the play mode, can be combination of 'self', 'bot', 'random', like 'self+bot'"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""

    total_timesteps: int = 1000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.997
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    update_win_rate: float = 0.6
    """the required win rate to update the agent"""
    update_return: float = 0.1
    """the required return to update the agent"""

    minibatch_size: int = 256
    """the mini-batch size"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    backend: Literal["gloo", "nccl", "mpi"] = "nccl"
    """the backend for distributed training"""

    compile: Optional[str] = None
    """Compile mode of torch.compile, None for no compilation"""
    torch_threads: Optional[int] = None
    """the number of threads to use for torch, defaults to ($OMP_NUM_THREADS or 2) * world_size"""
    env_threads: Optional[int] = None
    """the number of threads to use for envpool, defaults to `num_envs`"""
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
    port: int = 12356
    """the port to use for distributed training"""
    eval_episodes: int = 128
    """the number of episodes to evaluate the model"""

    # to be filled in runtime
    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
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


def run(local_rank, world_size):
    args = tyro.cli(Args)
    args.world_size = world_size
    args.local_num_envs = args.num_envs // args.world_size
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.minibatch_size // args.world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
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

        ckpt_dir = os.path.join(args.ckpt_dir, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)


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
        play_mode='self',
    )
    envs.num_envs = args.local_num_envs
    obs_space = envs.observation_space
    action_shape = envs.action_space.shape
    if local_rank == 0:
        print(f"obs_space={obs_space}, action_shape={action_shape}")

    envs_per_thread = args.local_num_envs // local_env_threads
    local_eval_episodes = args.eval_episodes // args.world_size
    local_eval_num_envs = local_eval_episodes
    eval_envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=local_eval_num_envs,
        num_threads=max(1, local_eval_num_envs // envs_per_thread),
        seed=args.seed,
        deck1=args.deck1,
        deck2=args.deck2,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        play_mode=args.play_mode,
    )
    eval_envs.num_envs = local_eval_num_envs

    envs = RecordEpisodeStatistics(envs)
    eval_envs = RecordEpisodeStatistics(eval_envs)

    if args.embedding_file:
        embeddings = np.load(args.embedding_file)
        embedding_shape = embeddings.shape
    else:
        embedding_shape = None
    L = args.num_layers
    agent1 = Agent(args.num_channels, L, L, 1, embedding_shape).to(device)
    if args.embedding_file:
        agent1.load_embeddings(embeddings)
    agent2 = Agent(args.num_channels, L, L, 1, embedding_shape).to(device)
    agent2.load_state_dict(agent1.state_dict())

    optim_params = list(agent1.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    def masked_mean(x, valid):
        x = x.masked_fill(~valid, 0)
        return x.sum() / valid.float().sum()

    def masked_normalize(x, valid, eps=1e-8):
        x = x.masked_fill(~valid, 0)
        n = valid.float().sum()
        mean = x.sum() / n
        var = ((x - mean) ** 2).sum() / n
        std = (var + eps).sqrt()
        return (x - mean) / std

    def train_step(agent: Agent, scaler, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_learns):
        with autocast(enabled=args.fp16_train):
            logits, newvalue, valid = agent(mb_obs)
            probs = Categorical(logits=logits)
            newlogprob = probs.log_prob(mb_actions)
            entropy = probs.entropy()
        valid = torch.logical_and(valid, mb_learns)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        if args.norm_adv:
            mb_advantages = masked_normalize(mb_advantages, valid, eps=1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = masked_mean(pg_loss, valid)

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
        v_loss = masked_mean(v_loss, valid)

        entropy_loss = masked_mean(entropy, valid)
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        return old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss

    def predict_step(agent1: Agent, agent2: Agent, next_obs, learn):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits1, value1, valid = agent1(next_obs)
                logits2, value2, valid = agent2(next_obs)
        logits = torch.where(learn[:, None], logits1, logits2)
        value = torch.where(learn[:, None], value1, value2)
        return logits, value

    def eval_step(agent: Agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits = agent.get_logit(next_obs)
        return logits

    if args.compile:
        train_step = torch.compile(train_step, mode=args.compile)
        predict_step = torch.compile(predict_step, mode='default')
        # eval_step = torch.compile(eval_step, mode=args.compile)

    def to_tensor(x, dtype=torch.float32):
        return optree.tree_map(lambda x: torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True), x)

    # ALGO Logic: Storage setup
    obs = create_obs(obs_space, (args.num_steps, args.local_num_envs), device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.local_num_envs), dtype=torch.bool).to(device)
    values = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    learns = torch.zeros((args.num_steps, args.local_num_envs), dtype=torch.bool).to(device)
    avg_ep_returns = deque(maxlen=1000)
    avg_win_rates = deque(maxlen=1000)
    version = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = to_tensor(next_obs, dtype=torch.uint8)
    next_to_play_ = info["to_play"]
    next_to_play = to_tensor(next_to_play_)
    next_done = torch.zeros(args.local_num_envs, device=device, dtype=torch.bool)
    ai_player_ = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(ai_player_)
    ai_player = to_tensor(ai_player_, dtype=next_to_play.dtype)
    next_value = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        model_time = 0
        env_time = 0
        collect_start = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs

            for key in obs:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            learn = next_to_play == ai_player
            learns[step] = learn

            _start = time.time()
            logits, value = predict_step(agent1, agent2, next_obs, learn)
            value = value.flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

            values[step] = value
            actions[step] = action
            logprobs[step] = logprob
            action = action.cpu().numpy()
            model_time += time.time() - _start

            next_value = torch.where(learn, value, next_value) * (1 - next_done.float())

            _start = time.time()
            to_play = next_to_play_
            next_obs, reward, next_done_, info = envs.step(action)
            next_to_play_ = info["to_play"]
            next_to_play = to_tensor(next_to_play_)
            env_time += time.time() - _start
            rewards[step] = to_tensor(reward)
            next_obs, next_done = to_tensor(next_obs, torch.uint8), to_tensor(next_done_, torch.bool)

            if not writer:
                continue

            for idx, d in enumerate(next_done_):
                if d:
                    pl = 1 if to_play[idx] == ai_player_[idx] else -1
                    episode_length = info['l'][idx]
                    episode_reward = info['r'][idx] * pl
                    win = 1 if episode_reward > 0 else 0
                    avg_ep_returns.append(episode_reward)
                    avg_win_rates.append(win)

                    if random.random() < args.log_p:
                        n = 100
                        if random.random() < 10/n or iteration <= 2:
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            print(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if random.random() < 1/n:
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_ep_returns), global_step)
                            writer.add_scalar("charts/avg_win_rate", np.mean(avg_win_rates), global_step)

        collect_time = time.time() - collect_start
        print(f"[Rank {local_rank}] collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}", flush=True)

        # bootstrap value if not done
        with torch.no_grad():
            value = agent1.get_value(next_obs).reshape(-1)
        advantages = torch.zeros_like(rewards).to(device)
        nextvalues = torch.where(next_to_play == ai_player, value, next_value)
        done_used = torch.zeros_like(next_done, dtype=torch.bool)
        reward = 0
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            # if learns[t]:
            #     if dones[t+1]:
            #         reward = rewards[t]
            #         nextvalues = 0
            #         lastgaelam = 0
            #         done_used = True
            #     else:
            #         if not done_used:
            #             reward = reward
            #             nextvalues = 0
            #             lastgaelam = 0
            #             done_used = True
            #         else:
            #             reward = rewards[t]
            #     delta = reward + args.gamma * nextvalues - values[t]
            #     lastgaelam_ = delta + args.gamma * args.gae_lambda * lastgaelam
            #     advantages[t] = lastgaelam_
            #     nextvalues = values[t]
            #     lastgaelam = lastgaelam_
            # else:
            #     if dones[t+1]:
            #         reward = -rewards[t]
            #         done_used = False
            #     else:
            #         reward = reward
            learn = learns[t]
            if t != args.num_steps - 1:
                next_done = dones[t + 1]
            sp = 2 * (learn.int() - 0.5)
            reward = torch.where(next_done, rewards[t] * sp, torch.where(learn & done_used, 0, reward))
            real_done = next_done | ~done_used
            nextvalues = torch.where(real_done, 0, nextvalues)
            lastgaelam = torch.where(real_done, 0, lastgaelam)
            done_used = torch.where(
                next_done, learn, torch.where(learn & ~done_used, True, done_used))

            delta = reward + args.gamma * nextvalues - values[t]
            advantages[t] = lastgaelam_ = delta + args.gamma * args.gae_lambda * lastgaelam
            nextvalues = torch.where(learn, values[t], nextvalues)
            lastgaelam = torch.where(learn, lastgaelam_, lastgaelam)
        returns = advantages + values

        _start = time.time()
        # flatten the batch
        b_obs = {
            k: v.reshape((-1,) + v.shape[2:])
            for k, v in obs.items()
        }
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_learns = learns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {
                    k: v[mb_inds] for k, v in b_obs.items()
                }
                old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                    train_step(agent1, scaler, mb_obs, b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds], b_learns[mb_inds])
                reduce_gradidents(optim_params, args.world_size)
                nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        train_time = time.time() - _start

        print(f"[Rank {local_rank}] train_time={train_time:.4f}, collect_time={collect_time:.4f}", flush=True)
        # if local_rank == 0:
        #     print(f"train_time={train_time:.4f}, collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if local_rank == 0:
            if iteration % args.save_interval == 0:
                torch.save(agent1.state_dict(), os.path.join(ckpt_dir, f"agent.pt"))

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            SPS = int((global_step - warmup_steps) / (time.time() - start_time))

            # Warmup at first few iterations for accurate SPS measurement
            SPS_warmup_iters = 10
            if iteration == SPS_warmup_iters:
                start_time = time.time()
                warmup_steps = global_step
            if iteration > SPS_warmup_iters:
                print("SPS:", SPS)
                writer.add_scalar("charts/SPS", SPS, global_step)

        if local_rank == 0:
            should_update = len(avg_win_rates) == 1000 and np.mean(avg_win_rates) > args.update_win_rate and np.mean(avg_ep_returns) > args.update_return
            should_update = torch.tensor(int(should_update), dtype=torch.int64, device=device)
        else:
            should_update = torch.zeros((), dtype=torch.int64, device=device)
        if args.world_size > 1:
            dist.all_reduce(should_update, op=dist.ReduceOp.SUM)
        should_update = should_update.item() > 0
        if should_update:
            agent2.load_state_dict(agent1.state_dict())
            version += 1
            if local_rank == 0:
                torch.save(agent1.state_dict(), os.path.join(ckpt_dir, f"agent_v{version}.pt"))
                print(f"Updating agent at global_step={global_step} with win_rate={np.mean(avg_win_rates)}")
                avg_win_rates.clear()
                avg_ep_returns.clear()

            _start = time.time()
            episode_lengths = []
            episode_rewards = []
            eval_win_rates = []
            e_obs = eval_envs.reset()[0]
            while True:
                e_obs = to_tensor(e_obs, dtype=torch.uint8)
                e_logits = eval_step(agent1, e_obs)
                e_probs = torch.softmax(e_logits, dim=-1)
                e_probs = e_probs.cpu().numpy()
                e_actions = e_probs.argmax(axis=1)

                e_obs, e_rewards, e_dones, e_info = eval_envs.step(e_actions)

                for idx, d in enumerate(e_dones):
                    if d:
                        episode_length = e_info['l'][idx]
                        episode_reward = e_info['r'][idx]
                        win = 1 if episode_reward > 0 else 0

                        episode_lengths.append(episode_length)
                        episode_rewards.append(episode_reward)
                        eval_win_rates.append(win)
                if len(episode_lengths) >= local_eval_episodes:
                    break
            
            eval_return = np.mean(episode_rewards[:local_eval_episodes])
            eval_ep_len = np.mean(episode_lengths[:local_eval_episodes])
            eval_win_rate = np.mean(eval_win_rates[:local_eval_episodes])
            eval_stats = torch.tensor([eval_return, eval_ep_len, eval_win_rate], dtype=torch.float32, device=device)

            # sync the statistics
            dist.all_reduce(eval_stats, op=dist.ReduceOp.AVG)
            if local_rank == 0:
                eval_return, eval_ep_len, eval_win_rate = eval_stats.cpu().numpy()
                writer.add_scalar("charts/eval_return", eval_return, global_step)
                writer.add_scalar("charts/eval_ep_len", eval_ep_len, global_step)
                writer.add_scalar("charts/eval_win_rate", eval_win_rate, global_step)
                eval_time = time.time() - _start
                print(f"eval_time={eval_time:.4f}, eval_ep_return={eval_return}, eval_ep_len={eval_ep_len}, eval_win_rate={eval_win_rate}")

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if local_rank == 0:
        torch.save(agent1.state_dict(), os.path.join(ckpt_dir, f"agent_final.pt"))
        writer.close()


if __name__ == "__main__":
    mp_start(run)
