import os
import random
import time
from dataclasses import dataclass
from typing import Optional


import ygoenv
import numpy as np
import tyro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics, to_tensor, load_embeddings
from ygoai.rl.agent import PPOAgent as Agent
from ygoai.rl.dist import reduce_gradidents, torchrun_setup, fprint
from ygoai.rl.buffer import create_obs
from ygoai.rl.ppo import bootstrap_value_selfplay, train_step as train_step_
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
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    update_win_rate: float = 0.55
    """the required win rate to update the agent"""
    self_play_prob: float = 0.6
    """the probability of self play"""
    max_lp: int = 6
    """the maximum number of LP to add model to the pool"""
    iter_per_lp: int = 1000
    """the number of iterations per learning phase"""
    target_sample_iter: int = 10
    """the number of iterations to sample the target model"""

    minibatch_size: int = 256
    """the mini-batch size"""
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
    collect_length: Optional[int] = None
    """the length of the buffer, only the first `num_steps` will be used for training (partial GAE)"""

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
    eval_episodes: int = 128
    """the number of episodes to evaluate the model"""
    eval_interval: int = 50
    """the number of iterations to evaluate the model"""

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
    num_embeddings: Optional[int] = None
    """the number of embeddings (computed in runtime)"""


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
        play_mode=mode,
    )
    envs.num_envs = num_envs
    envs = RecordEpisodeStatistics(envs)
    return envs


def update_running_mean(mean, value, count):
    return mean + (value - mean) / count

def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}")

    args = tyro.cli(Args)
    args.world_size = world_size
    args.local_num_envs = args.num_envs // args.world_size
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.minibatch_size // args.world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_minibatches = args.local_batch_size // args.local_minibatch_size
    args.env_threads = args.env_threads or args.num_envs
    args.torch_threads = args.torch_threads or (int(os.getenv("OMP_NUM_THREADS", "2")) * args.world_size)
    args.collect_length = args.collect_length or args.num_steps

    assert args.collect_length >= args.num_steps, "collect_length must be greater than or equal to num_steps"

    local_torch_threads = args.torch_threads // args.world_size
    local_env_threads = args.env_threads // args.world_size

    torch.set_num_threads(local_torch_threads)
    torch.set_float32_matmul_precision('high')

    if args.world_size > 1:
        torchrun_setup('nccl', local_rank)

    def sync_var(var, dtype=torch.float32, reduce='first'):
        ts = torch.tensor(var, dtype=dtype, device=device)
        if reduce == 'mean':
            if args.world_size > 1:
                dist.all_reduce(ts, op=dist.ReduceOp.AVG)
        else:
            if rank != 0:
                ts = torch.zeros_like(ts)
            if args.world_size > 1:
                dist.all_reduce(ts, op=dist.ReduceOp.SUM)
        return ts.cpu().numpy()

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    writer = None
    if rank == 0:
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
    args.seed += rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - rank)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    deck = init_ygopro(args.env_id, "english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, args.local_num_envs, local_env_threads)
    obs_space = envs.env.observation_space
    action_shape = envs.env.action_space.shape
    if local_rank == 0:
        fprint(f"obs_space={obs_space}, action_shape={action_shape}")

    envs_per_thread = args.local_num_envs // local_env_threads
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
    torch.manual_seed(args.seed)
    agent.eval()

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        fprint(f"Loaded checkpoint from {args.checkpoint}")
    elif args.embedding_file:
        agent.load_embeddings(embeddings)
        fprint(f"Loaded embeddings from {args.embedding_file}")
    if args.embedding_file:
        agent.freeze_embeddings()

    agent_t = Agent(args.num_channels, L, L, embedding_shape).to(device)
    agent_t.eval()
    agent_t.load_state_dict(agent.state_dict())

    optim_params = list(agent.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    def predict_step(agent: Agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits, value, valid = agent(next_obs)
        return logits, value

    history = []

    if args.compile:
        # It seems that using torch.compile twice cause segfault at start, so we use torch.jit.trace here
        # predict_step = torch.compile(predict_step, mode=args.compile)
        example_obs = create_obs(envs.observation_space, (args.local_num_envs,), device=device)
        with torch.no_grad():
            traced_model = torch.jit.trace(agent, (example_obs,), check_tolerance=False, check_trace=False)
        if args.checkpoint:
            with torch.no_grad():
                traced_model_t = torch.jit.trace(agent_t, (example_obs,), check_tolerance=False, check_trace=False)
            traced_model_t = torch.jit.optimize_for_inference(traced_model_t)
            history.append(traced_model_t)

        train_step = torch.compile(train_step_, mode=args.compile)
    else:
        train_step = train_step_


    def sample_target(history):
        ts = []
        for i in range(args.target_sample_iter):
            if len(history) == 0 or random.random() < args.self_play_prob:
                ts.append(-1)
            else:
                ts.append(random.randint(0, len(history) - 1))
        ts.sort(reverse=True)
        return sync_var(ts, dtype=torch.int64).tolist()

    # ALGO Logic: Storage setup
    obs = create_obs(obs_space, (args.collect_length, args.local_num_envs), device)
    actions = torch.zeros((args.collect_length, args.local_num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.collect_length, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.collect_length, args.local_num_envs)).to(device)
    dones = torch.zeros((args.collect_length, args.local_num_envs), dtype=torch.bool).to(device)
    values = torch.zeros((args.collect_length, args.local_num_envs)).to(device)
    learns = torch.zeros((args.collect_length, args.local_num_envs), dtype=torch.bool).to(device)
    avg_ep_returns = [0]
    avg_win_rates = [0]
    n_episodes = [0]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_done = torch.zeros(args.local_num_envs, device=device, dtype=torch.bool)
    main_player_ = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(main_player_)
    main_player = to_tensor(main_player_, device)
    next_value1 = next_value2 = 0
    step = 0
    ts = []
    lp_count = 0

    for iteration in range(args.num_iterations):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration % args.iter_per_lp)  / args.iter_per_lp
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        if iteration % args.iter_per_lp == 0:
            next_obs, info = envs.reset()
            next_obs = to_tensor(next_obs, device, dtype=torch.uint8)
            next_to_play_ = info["to_play"]
            next_to_play = to_tensor(next_to_play_, device)
            next_value1 = next_value2 = 0
            step = 0
            ts = []

        if len(ts) == 0:
            ts = sample_target(history)
        t_idx = ts.pop()
        selfplay = t_idx == -1
        if not selfplay:
            traced_model_t = history[t_idx]

        model_time = 0
        env_time = 0
        collect_start = time.time()
        while step < args.collect_length:
            global_step += args.num_envs

            for key in obs:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            learn = next_to_play == main_player
            learns[step] = learn

            _start = time.time()
            logits, value = predict_step(traced_model, next_obs)
            if not selfplay:
                logits_t, value_t = predict_step(traced_model_t, next_obs)
                logits = torch.where(learn[:, None], logits, logits_t)
                value = torch.where(learn[:, None], value, value_t)
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

            for idx, d in enumerate(next_done_):
                if d:
                    pl = 1 if to_play[idx] == main_player_[idx] else -1
                    episode_length = info['l'][idx]
                    episode_reward = info['r'][idx] * pl
                    win = 1 if episode_reward > 0 else 0
                    if len(history) == 0 or not selfplay:
                        n_episodes[t_idx] += 1
                        avg_ep_returns[t_idx] = update_running_mean(avg_ep_returns[t_idx], episode_reward, n_episodes[t_idx])
                        avg_win_rates[t_idx] = update_running_mean(avg_win_rates[t_idx], win, n_episodes[t_idx])

                    if writer and random.random() < args.log_p:
                        n = 100
                        if random.random() < 10/n or iteration <= 1:
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            fprint(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if random.random() < 1/n:
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_ep_returns), global_step)
                            writer.add_scalar("charts/avg_win_rate", np.mean(avg_win_rates), global_step)

        collect_time = time.time() - collect_start
        if local_rank == 0:
            fprint(f"collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        step = args.collect_length - args.num_steps

        _start = time.time()
        # bootstrap value if not done
        with torch.no_grad():
            value = predict_step(traced_model, next_obs)[1].reshape(-1)
            if not selfplay:
                value_t = predict_step(traced_model_t, next_obs)[1].reshape(-1)
                value = torch.where(next_to_play == main_player, value, value_t)
        nextvalues1 = torch.where(next_to_play == main_player, value, next_value1)
        nextvalues2 = torch.where(next_to_play != main_player, value, next_value2)

        if step > 0 and iteration != 0:
            # recalculate the values for the first few steps
            v_steps = args.local_minibatch_size * 4 // args.local_num_envs
            for v_start in range(0, step, v_steps):
                v_end = min(v_start + v_steps, step)
                v_obs = {
                    k: v[v_start:v_end].flatten(0, 1) for k, v in obs.items()
                }
                with torch.no_grad():
                    # value = traced_get_value(v_obs).reshape(v_end - v_start, -1)
                    value = predict_step(traced_model, v_obs)[1].reshape(v_end - v_start, -1)
                values[v_start:v_end] = value

        advantages = bootstrap_value_selfplay(
            values, rewards, dones, learns, nextvalues1, nextvalues2, next_done, args.gamma, args.gae_lambda)
        bootstrap_time = time.time() - _start

        _start = time.time()
        # flatten the batch
        b_obs = {
            k: v[:args.num_steps].reshape((-1,) + v.shape[2:])
            for k, v in obs.items()
        }
        b_actions = actions[:args.num_steps].reshape((-1,) + action_shape)
        b_logprobs = logprobs[:args.num_steps].reshape(-1)
        b_advantages = advantages[:args.num_steps].reshape(-1)
        b_values = values[:args.num_steps].reshape(-1)
        b_returns = b_advantages + b_values
        if selfplay:
            b_learns = torch.ones_like(b_values, dtype=torch.bool)
        else:
            b_learns = learns[:args.num_steps].reshape(-1)

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
                    train_step(agent, optimizer, scaler, mb_obs, b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds], b_learns[mb_inds], args)
                reduce_gradidents(optim_params, args.world_size)
                nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

        if step > 0:
            # TODO: use cyclic buffer to avoid copying
            for v in obs.values():
                v[:step] = v[args.num_steps:].clone()
            for v in [actions, logprobs, rewards, dones, values, learns]:
                v[:step] = v[args.num_steps:].clone()

        train_time = time.time() - _start

        if local_rank == 0:
            fprint(f"train_time={train_time:.4f}, collect_time={collect_time:.4f}, bootstrap_time={bootstrap_time:.4f}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if rank == 0:
            if iteration % args.save_interval == 0:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent.pt"))

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
            if local_rank == 0:
                fprint(f"SPS: {SPS}")
            if rank == 0:
                writer.add_scalar("charts/SPS", SPS, global_step)

        if (iteration + 1) % args.iter_per_lp == 0:
            lp_count += 1
            win_rates = sync_var(avg_win_rates, dtype=torch.float32, reduce='mean')
            if len(history) == 0 or np.all(win_rates > args.update_win_rate) or lp_count >= args.max_lp:
                agent_t.load_state_dict(agent.state_dict())
                with torch.no_grad():
                    traced_model_t = torch.jit.trace(agent_t, (example_obs,), check_tolerance=False, check_trace=False)
                traced_model_t = torch.jit.optimize_for_inference(traced_model_t)
                history.append(traced_model_t)
                lp_count = 0
                if rank == 0:
                    version = len(history)
                    torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_v{version}.pt"))
                    fprint(f"model v{version} added to the pool, win_rates={win_rates}")
            else:
                if rank == 0:
                    fprint(f"win_rates={win_rates}, not updating the pool")
            avg_ep_returns = [0] * len(history)
            avg_win_rates = [0] * len(history)
            n_episodes = [0] * len(history)

        if args.eval_interval and iteration % args.eval_interval == 0:
            # Eval with rule-based policy
            _start = time.time()
            eval_return = evaluate(
                eval_envs, traced_model, local_eval_episodes, device, args.fp16_eval)[0]
            eval_stats = torch.tensor(eval_return, dtype=torch.float32, device=device)

            # sync the statistics
            if args.world_size > 1:
                dist.all_reduce(eval_stats, op=dist.ReduceOp.AVG)
            eval_return = eval_stats.cpu().numpy()
            if rank == 0:
                writer.add_scalar("charts/eval_return", eval_return, global_step)
            if local_rank == 0:
                eval_time = time.time() - _start
                fprint(f"eval_time={eval_time:.4f}, eval_ep_return={eval_return:.4f}")

            # Eval with old model

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if rank == 0:
        torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_final.pt"))
        writer.close()


if __name__ == "__main__":
    main()
