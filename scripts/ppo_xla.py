import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


import ygoenv
import numpy as np
import tyro

import torch
torch.set_num_threads(2)
import torch.optim as optim
import torch.distributed as dist

import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics, to_tensor, load_embeddings
from ygoai.rl.agent import PPOAgent as Agent
from ygoai.rl.dist import fprint
from ygoai.rl.buffer import create_obs, get_obs_shape
from ygoai.rl.ppo import bootstrap_value_selfplay_np as bootstrap_value_selfplay
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
    local_num_envs: int = 256
    "the number of parallel game environments"
    local_env_threads: Optional[int] = None
    "the number of threads to use for environment"
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    fix_target: bool = False
    """if toggled, the target network will be fixed"""
    update_win_rate: float = 0.55
    """the required win rate to update the agent"""
    update_return: float = 0.1
    """the required return to update the agent"""

    local_minibatch_size: int = 4096
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
    minibatch_size: int = 0
    num_envs: int = 0
    batch_size: int = 0
    num_iterations: int = 0
    world_size: int = 0
    num_embeddings: Optional[int] = None


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

def _mp_fn(index, world_size):
    rank = index
    local_rank = index
    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}")

    args = tyro.cli(Args)
    args.world_size = world_size
    args.num_envs = args.local_num_envs * args.world_size
    args.local_batch_size = args.local_num_envs * args.num_steps
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    args.local_env_threads = args.local_env_threads or args.local_num_envs
    args.env_threads = args.local_env_threads * args.world_size
    args.torch_threads = args.torch_threads or (int(os.getenv("OMP_NUM_THREADS", "2")) * args.world_size)
    args.collect_length = args.collect_length or args.num_steps

    assert args.local_batch_size % args.local_minibatch_size == 0, "local_batch_size must be divisible by local_minibatch_size"
    assert args.collect_length >= args.num_steps, "collect_length must be greater than or equal to num_steps"

    torch.set_num_threads(2)
    # torch.set_float32_matmul_precision('high')

    if args.world_size > 1:
        dist.init_process_group('xla', init_method='xla://')

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
    # if args.torch_deterministic:
    #     torch.backends.cudnn.deterministic = True
    # else:
    #     torch.backends.cudnn.benchmark = True

    device = xm.xla_device()

    deck = init_ygopro(args.env_id, "english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, args.local_num_envs, args.local_env_threads)
    obs_space = envs.env.observation_space
    action_shape = envs.env.action_space.shape
    if local_rank == 0:
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

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        fprint(f"Loaded checkpoint from {args.checkpoint}")
    elif args.embedding_file:
        agent.load_embeddings(embeddings)
        fprint(f"Loaded embeddings from {args.embedding_file}")
    if args.embedding_file:
        agent.freeze_embeddings()

    if args.fix_target:
        agent_t = Agent(args.num_channels, L, L, embedding_shape).to(device)
        agent_t.eval()
        agent_t.load_state_dict(agent.state_dict())
    else:
        agent_t = agent

    # if args.world_size > 1:
    #     ddp_agent = DDP(agent, gradient_as_bucket_view=True)
    # else:
    #     ddp_agent = agent

    optim_params = list(agent.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    def predict_step(agent: Agent, next_obs):
        with torch.no_grad():
            logits, value, valid = agent(next_obs)
        return logits, value

    from ygoai.rl.ppo import train_step_t as train_step
    if args.compile:
        traced_model = torch.compile(agent, backend='openxla_eval')
        traced_model_t = traced_model
        train_step = torch.compile(train_step, backend='openxla')
    else:
        traced_model = agent
        traced_model_t = agent_t

    # ALGO Logic: Storage setup
    obs_shape = get_obs_shape(obs_space)
    obs = {
        key: np.zeros(
            (args.collect_length, args.local_num_envs, *_obs_shape), dtype=obs_space[key].dtype)
        for key, _obs_shape in obs_shape.items()
    }
    actions = np.zeros((args.collect_length, args.local_num_envs) + action_shape, dtype=np.int64)
    logprobs = np.zeros((args.collect_length, args.local_num_envs), dtype=np.float32)
    rewards = np.zeros((args.collect_length, args.local_num_envs), dtype=np.float32)
    dones = np.zeros((args.collect_length, args.local_num_envs), dtype=np.bool_)
    values = np.zeros((args.collect_length, args.local_num_envs), dtype=np.float32)
    learns = np.zeros((args.collect_length, args.local_num_envs), dtype=np.bool_)
    avg_ep_returns = deque(maxlen=1000)
    avg_win_rates = deque(maxlen=1000)
    version = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs_ = to_tensor(next_obs, device, dtype=torch.uint8)
    next_to_play = info["to_play"]
    next_done = np.zeros(args.local_num_envs, dtype=np.bool_)
    ai_player1 = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(ai_player1)
    next_value1 = next_value2 = 0
    step = 0

    for iteration in range(args.num_iterations):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - iteration / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        model_time = 0
        env_time = 0
        o_time1 = 0
        o_time2 = 0
        collect_start = time.time()
        while step < args.collect_length:
            global_step += args.num_envs

            _start = time.time()
            for key in obs:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            learn = next_to_play == ai_player1
            learns[step] = learn
            o_time1 += time.time() - _start

            _start = time.time()
            logits, value = predict_step(traced_model, next_obs_)
            if args.fix_target:
                logits_t, value_t = predict_step(traced_model_t, next_obs)
                logits = torch.where(learn[:, None], logits, logits_t)
                value = torch.where(learn[:, None], value, value_t)
            u = torch.rand_like(logits)
            action = torch.argmax(logits - torch.log(-torch.log(u)), dim=1)
            logprob = logits.log_softmax(dim=1).gather(-1, action[:, None]).squeeze(-1)
            value = value.flatten()
            xm.mark_step()
            model_time += time.time() - _start

            _start = time.time()
            logprob = logprob.cpu().numpy()
            value = value.cpu().numpy()
            action = action.cpu().numpy()
            o_time2 += time.time() - _start
            
            _start = time.time()
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_nonterminal = 1 - next_done.astype(np.float32)
            next_value1 = np.where(learn, value, next_value1) * next_nonterminal
            next_value2 = np.where(learn, next_value2, value) * next_nonterminal
            o_time1 += time.time() - _start

            _start = time.time()
            to_play = next_to_play
            next_obs, reward, next_done, info = envs.step(action)
            next_to_play = info["to_play"]
            env_time += time.time() - _start
            _start = time.time()
            rewards[step] = reward
            o_time1 += time.time() - _start
            
            _start = time.time()
            next_obs_ = to_tensor(next_obs, device, torch.uint8)
            o_time2 += time.time() - _start
            step += 1

            if not writer:
                continue

            for idx, d in enumerate(next_done):
                if d:
                    pl = 1 if to_play[idx] == ai_player1[idx] else -1
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
        # if local_rank == 0:
        fprint(f"[Rank {rank}] collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}, o_time1={o_time1:.4f}, o_time2={o_time2:.4f}")

        step = args.collect_length - args.num_steps

        _start = time.time()
        # bootstrap value if not done
        with torch.no_grad():
            value = predict_step(traced_model, next_obs_)[1].reshape(-1)
            if args.fix_target:
                value_t = predict_step(traced_model_t, next_obs_)[1].reshape(-1)
                value = torch.where(next_to_play == ai_player1, value, value_t)
            value = value.cpu().numpy()
        nextvalues1 = np.where(next_to_play == ai_player1, value, next_value1)
        nextvalues2 = np.where(next_to_play != ai_player1, value, next_value2)

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

        train_start = time.time()
        d_time1 = 0
        d_time2 = 0
        d_time3 = 0
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
        if args.fix_target:
            b_learns = learns[:args.num_steps].reshape(-1)
        else:
            b_learns = np.ones_like(b_values, dtype=np.bool_)

        _start = time.time()
        b_obs = to_tensor(b_obs, device=device, dtype=torch.uint8)
        b_actions, b_logprobs, b_advantages, b_values, b_returns, b_learns = [
            to_tensor(v, device) for v in [b_actions, b_logprobs, b_advantages, b_values, b_returns, b_learns]
        ]
        d_time1 += time.time() - _start

        agent.train()
        
        model_time = 0

        # Optimizing the policy and value network
        clipfracs = []
        b_inds = np.arange(args.local_batch_size)
        xm.mark_step()
        for epoch in range(args.update_epochs):
            _start = time.time()
            np.random.shuffle(b_inds)
            d_time2 += time.time() - _start

            _start = time.time()
            b_inds_ = to_tensor(b_inds, device=device)
            n_mini_batches = args.local_batch_size // args.local_minibatch_size
            b_inds_ = b_inds_.reshape(n_mini_batches, args.local_minibatch_size)
            xm.mark_step()
            d_time3 += time.time() - _start
            for i in range(n_mini_batches):
                _start = time.time()
                mb_inds = b_inds_[i]
                xm.mark_step()
                d_time3 += time.time() - _start

                _start = time.time()
                old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                    train_step(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages,
                               b_returns, b_values, b_learns, mb_inds, args)
                clipfracs.append(clipfrac)
                xm.mark_step()
                model_time += time.time() - _start

                # mb_obs = {
                #     k: v[mb_inds] for k, v in b_obs.items()
                # }
                # mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_learns = [
                #     v[mb_inds] for v in [b_actions, b_logprobs, b_advantages, b_returns, b_values, b_learns]]
                # xm.mark_step()
                # old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                #     train_step(ddp_agent_t, optimizer, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                #                mb_returns, mb_values, mb_learns, args)
                
                # if rank == 0:
                #     # For short report that only contains a few key metrics.
                #     print(met.short_metrics_report())
                #     # For full report that includes all metrics.
                #     print(met.metrics_report())
                #     met.clear_all()

        clipfrac = torch.stack(clipfracs).mean().item()
        
        if step > 0:
            # TODO: use cyclic buffer to avoid copying
            for v in obs.values():
                v[:step] = v[args.num_steps:].clone()
            for v in [actions, logprobs, rewards, dones, values, learns]:
                v[:step] = v[args.num_steps:].clone()

        train_time = time.time() - train_start

        if local_rank == 0:
            fprint(f"d_time1={d_time1:.4f}, d_time2={d_time2:.4f}, d_time3={d_time3:.4f}")
            fprint(f"train_time={train_time:.4f}, model_time={model_time:.4f}, collect_time={collect_time:.4f}, bootstrap_time={bootstrap_time:.4f}")

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
            writer.add_scalar("losses/clipfrac", clipfrac, global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

        SPS = int((global_step - warmup_steps) / (time.time() - start_time))

        # Warmup at first few iterations for accurate SPS measurement
        SPS_warmup_iters = 5
        if iteration == SPS_warmup_iters:
            start_time = time.time()
            warmup_steps = global_step
        if iteration > SPS_warmup_iters:
            if local_rank == 0:
                fprint(f"SPS: {SPS}")
            if rank == 0:
                writer.add_scalar("charts/SPS", SPS, global_step)

        if args.fix_target:
            if rank == 0:
                should_update = len(avg_win_rates) == 1000 and np.mean(avg_win_rates) > args.update_win_rate and np.mean(avg_ep_returns) > args.update_return
                should_update = torch.tensor(int(should_update), dtype=torch.int64, device=device)
            else:
                should_update = torch.zeros((), dtype=torch.int64, device=device)
            if args.world_size > 1:
                dist.all_reduce(should_update, op=dist.ReduceOp.SUM)
            should_update = should_update.item() > 0
            if should_update:
                agent_t.load_state_dict(agent.state_dict())
                with torch.no_grad():
                    traced_model_t = torch.jit.trace(agent_t, (example_obs,), check_tolerance=False, check_trace=False)
                    traced_model_t = torch.jit.optimize_for_inference(traced_model_t)

                version += 1
                if rank == 0:
                    torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_v{version}.pt"))
                    print(f"Updating agent at global_step={global_step} with win_rate={np.mean(avg_win_rates)}")
                    avg_win_rates.clear()
                    avg_ep_returns.clear()

        # if args.eval_interval and iteration % args.eval_interval == 0:
        #     # Eval with rule-based policy
        #     _start = time.time()
        #     eval_return = evaluate(
        #         eval_envs, traced_model, local_eval_episodes, device, args.fp16_eval)[0]
        #     eval_stats = torch.tensor(eval_return, dtype=torch.float32, device=device)

        #     # sync the statistics
        #     if args.world_size > 1:
        #         dist.all_reduce(eval_stats, op=dist.ReduceOp.AVG)
        #     eval_return = eval_stats.cpu().numpy()
        #     if rank == 0:
        #         writer.add_scalar("charts/eval_return", eval_return, global_step)
        #     if local_rank == 0:
        #         eval_time = time.time() - _start
        #         fprint(f"eval_time={eval_time:.4f}, eval_ep_return={eval_return:.4f}")

            # Eval with old model

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if rank == 0:
        torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_final.pt"))
        writer.close()


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        _mp_fn(0, 1)
    else:
        xmp.spawn(_mp_fn, args=(world_size,))
