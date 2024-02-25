import os
import random
import time
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
    play_mode: str = "self"
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
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
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

    compile: bool = True
    """whether to use torch.compile to compile the model and functions"""
    compile_mode: Optional[str] = None
    """the mode to use for torch.compile"""
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
        play_mode=args.play_mode,
    )
    envs.num_envs = args.local_num_envs
    obs_space = envs.observation_space
    action_shape = envs.action_space.shape
    if local_rank == 0:
        print(f"obs_space={obs_space}, action_shape={action_shape}")

    envs = RecordEpisodeStatistics(envs)

    if args.embedding_file:
        embeddings = np.load(args.embedding_file)
        embedding_shape = embeddings.shape
    else:
        embedding_shape = None
    L = args.num_layers
    agent = Agent(args.num_channels, L, L, 1, embedding_shape).to(device)
    if args.embedding_file:
        agent.load_embeddings(embeddings)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    def masked_mean(x, valid):
        x = x.masked_fill(~valid, 0)
        return x.sum() / valid.float().sum()

    def train_step(agent, scaler, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values):
        with autocast(enabled=args.fp16_train):
            _, newlogprob, entropy, newvalue, valid = agent.get_action_and_value(mb_obs, mb_actions.long())
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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
        reduce_gradidents(agent, args.world_size)
        return old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss

    def predict_step(agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits, values = agent(next_obs)
        return logits, values

    if args.compile:
        train_step = torch.compile(train_step, mode=args.compile_mode)
        predict_step = torch.compile(predict_step, mode=args.compile_mode)

    def to_tensor(x, dtype=torch.float32):
        return optree.tree_map(lambda x: torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True), x)

    # ALGO Logic: Storage setup
    obs = create_obs(obs_space, (args.num_steps, args.local_num_envs), device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    avg_ep_returns = []
    avg_win_rates = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs = to_tensor(envs.reset()[0], dtype=torch.uint8)
    next_done = torch.zeros(args.local_num_envs, device=device)

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

            _start = time.time()
            logits, value = predict_step(agent, next_obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            action = action.cpu().numpy()
            model_time += time.time() - _start

            _start = time.time()
            next_obs, reward, next_done_, info = envs.step(action)
            env_time += time.time() - _start
            rewards[step] = to_tensor(reward)
            next_obs, next_done = to_tensor(next_obs, torch.uint8), to_tensor(next_done_)

            if not writer:
                continue

            for idx, d in enumerate(next_done_):
                if d:
                    episode_length = info['l'][idx]
                    episode_reward = info['r'][idx]
                    winner = 0 if episode_reward > 0 else 1
                    avg_ep_returns.append(episode_reward)
                    avg_win_rates.append(1 - winner)

                    if random.random() < args.log_p:
                        n = 100
                        if random.random() < 10/n or iteration <= 2:
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            print(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if len(avg_win_rates) > n:
                            writer.add_scalar("charts/avg_win_rate", np.mean(avg_win_rates), global_step)
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_ep_returns), global_step)
                            avg_win_rates = []
                            avg_ep_returns = []

        collect_time = time.time() - collect_start

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
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
                    train_step(agent, scaler, mb_obs, b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds])
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        train_time = time.time() - _start

        if local_rank == 0:
            print(f"train_time={train_time:.4f}, collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if local_rank == 0:
            if iteration % args.save_interval == 0 or iteration == args.num_iterations:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent.pth"))

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

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if local_rank == 0:
        writer.close()


if __name__ == "__main__":
    mp_start(run)
