import os
import shutil
import queue
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from collections import deque
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Literal
from functools import partial

import ygoenv
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import tyro
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

from ygoai.utils import init_ygopro, load_embeddings
from ygoai.rl.ckpt import ModelCheckpoint, sync_to_gcs, zip_files
from ygoai.rl.jax.agent import RNNAgent, ModelArgs, RNDModel, EncoderArgs, default_rnd_args
from ygoai.rl.jax.utils import RecordEpisodeStatistics, masked_normalize, categorical_sample, RunningMeanStd
from ygoai.rl.jax.eval import evaluate, battle
from ygoai.rl.jax import clipped_surrogate_pg_loss, vtrace_2p0s, mse_loss, entropy_loss, \
    simple_policy_loss, ach_loss, policy_gradient_loss, truncated_gae
from ygoai.rl.jax.switch import truncated_gae_2p0s as gae_2p0s_switch


os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    log_frequency: int = 10
    """the logging frequency of the model performance (in terms of `updates`)"""
    time_log_freq: int = 0
    """the logging frequency of the deck time statistics, 0 to disable"""
    save_interval: int = 400
    """the frequency of saving the model (in terms of `updates`)"""
    checkpoint: Optional[str] = None
    """the path to the model checkpoint to load"""
    timeout: int = 600
    """the timeout of the environment step"""
    debug: bool = False
    """whether to run the script in debug mode"""

    tb_dir: str = "runs"
    """the directory to save the tensorboard logs"""
    tb_offset: int = 0
    """the step offset of the tensorboard logs"""
    run_name: Optional[str] = None
    """the name of the tensorboard run"""
    ckpt_dir: str = "checkpoints"
    """the directory to save the model checkpoints"""
    gcs_bucket: Optional[str] = None
    """the GCS bucket to save the model checkpoints"""

    # Algorithm specific arguments
    env_id: str = "YGOPro-v1"
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
    greedy_reward: bool = False
    """whether to use greedy reward (faster kill higher reward)"""

    total_timesteps: int = 50000000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 128
    """the number of parallel game environments"""
    local_env_threads: Optional[int] = None
    """the number of threads to use for environment"""
    num_actor_threads: int = 2
    """the number of actor threads to use"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    segment_length: Optional[int] = None
    """the length of the segment for training"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    switch: bool = False
    """Toggle the use of switch mechanism"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    burn_in_steps: Optional[int] = None
    """the number of burn-in steps for training (for R2D2)"""

    upgo: bool = True
    """Toggle the use of UPGO for advantages"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    c_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    c_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""
    rho_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    rho_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""

    ppo_clip: bool = True
    """whether to use the PPO clipping to replace V-Trace surrogate clipping"""
    clip_coef: float = 0.25
    """the PPO surrogate clipping coefficient"""
    dual_clip_coef: Optional[float] = 3.0
    """the dual surrogate clipping coefficient, typically 3.0"""
    spo_kld_max: Optional[float] = None
    """the maximum KLD for the SPO policy, typically 0.02"""
    logits_threshold: Optional[float] = None
    """the logits threshold for NeuRD and ACH, typically 2.0-6.0"""

    vloss_clip: Optional[float] = None
    """the value loss clipping coefficient"""

    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    m1: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent"""
    m2: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the eval agent"""

    mr: EncoderArgs = field(default_factory=lambda: default_rnd_args)
    """the model arguments for the RND network"""
    int_gamma: float = 0.99
    """the gamma for the intrinsic reward"""
    rnd_update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    rnd_episodic: bool = False
    """whether to use episodic intrinsic reward for RND"""
    rnd_norm: Literal["default", "min_max", "min_max2"] = "default"
    """the normalization method for RND intrinsic reward"""
    int_coef: float = 0.5
    """coefficient of intrinsic reward, 0.0 to disable RND"""
    ext_coef: float = 1.0
    """coefficient of extrinsic reward"""
    reward_scale: float = 1.0
    """the scaling factor of the reward"""

    actor_device_ids: List[int] = field(default_factory=lambda: [0, 1])
    """the device ids that actor workers will use"""
    learner_device_ids: List[int] = field(default_factory=lambda: [2, 3])
    """the device ids that learner workers will use"""
    distributed: bool = False
    """whether to use `jax.distirbuted`"""
    concurrency: bool = True
    """whether to run the actor and learner concurrently"""
    bfloat16: bool = False
    """whether to use bfloat16 for the agent"""
    thread_affinity: bool = False
    """whether to use thread affinity for the environment"""

    eval_checkpoint: Optional[str] = None
    """the path to the model checkpoint to evaluate"""
    local_eval_episodes: int = 128
    """the number of episodes to evaluate the model"""
    eval_interval: int = 100
    """the number of iterations to evaluate the model"""

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0
    global_learner_decices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    learner_devices: Optional[List[str]] = None
    num_embeddings: Optional[int] = None
    freeze_id: Optional[bool] = None
    deck_names: Optional[List[str]] = None
    real_seed: Optional[int] = None
    enable_rnd: Optional[bool] = None


def make_env(args, seed, num_envs, num_threads, mode='self', thread_affinity_offset=-1, eval=False):
    if not args.thread_affinity:
        thread_affinity_offset = -1
    if thread_affinity_offset >= 0:
        print("Binding to thread offset", thread_affinity_offset)
    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=num_threads,
        thread_affinity_offset=thread_affinity_offset,
        seed=seed,
        deck1=args.deck1,
        deck2=args.deck2,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        async_reset=False,
        greedy_reward=args.greedy_reward if not eval else True,
        play_mode=mode,
        timeout=args.timeout,
    )
    envs.num_envs = num_envs
    return envs


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logits: list
    rewards: list
    mains: list
    next_dones: list
    target_feats: list = None
    int_rewards: list = None


def create_agent(args, eval=False):
    if eval:
        return RNNAgent(
            embedding_shape=args.num_embeddings,
            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
            param_dtype=jnp.float32,
            **asdict(args.m2),
        )
    else:
        return RNNAgent(
            embedding_shape=args.num_embeddings,
            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
            param_dtype=jnp.float32,
            switch=args.switch,
            freeze_id=args.freeze_id,
            int_head=args.enable_rnd,
            **asdict(args.m1),
        )


def init_rnn_state(num_envs, rnn_channels):
    return (
        np.zeros((num_envs, rnn_channels)),
        np.zeros((num_envs, rnn_channels)),
    )


def create_rnd_model(args, predictor=False):
    return RNDModel(
        is_predictor=predictor,
        embedding_shape=args.num_embeddings,
        dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
        param_dtype=jnp.float32,
        freeze_id=args.freeze_id,
        **asdict(args.mr),
    )


def reshape_minibatch(
    x, multi_step, num_minibatches, num_steps, segment_length=None, key=None):
    # if segment_length is None,
    #   n_mb = num_minibatches
    #   if multi_step, from (num_steps, num_envs, ...)) to
    #     (n_mb, num_steps * (num_envs // n_mb), ...)
    #   else, from (num_envs, ...) to
    #     (n_mb, num_envs // n_mb, ...)
    # else,
    #   n_mb_t = num_steps // segment_length
    #   n_mb_e = num_minibatches // n_mb_t
    #   if multi_step, from (num_steps, num_envs, ...)) to
    #     (n_mb_e, n_mb_t, segment_length * (num_envs // n_mb_e), ...)
    #   else, from (num_envs, ...) to
    #     (n_mb_e, num_envs // n_mb_e, ...)
    if key is not None:
        x = jax.random.permutation(key, x, axis=1 if multi_step else 0)

    N = num_minibatches
    if segment_length is None:
        if multi_step:
            x = jnp.reshape(x, (num_steps, N, -1) + x.shape[2:])
            x = x.transpose(1, 0, *range(2, x.ndim))
            x = x.reshape(N, -1, *x.shape[3:])
        else:
            x = jnp.reshape(x, (N, -1) + x.shape[1:])
    else:
        M = segment_length
        Nt = num_steps // M
        Ne = N // Nt
        if multi_step:
            x = jnp.reshape(x, (Nt, M, Ne, -1) + x.shape[2:])
            x = x.transpose(2, 0, 1, *range(3, x.ndim))
            x = jnp.reshape(x, (Ne, Nt, -1) + x.shape[4:])
        else:
            x = jnp.reshape(x, (Ne, -1) + x.shape[1:])
    return x


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue,
    writer,
    actor_device,
    learner_devices,
    device_thread_id,
):
    eval_mode = 'self' if args.eval_checkpoint else 'bot'
    if eval_mode != 'bot':
        params_rt, eval_params = params_queue.get()
    else:
        params_rt, = params_queue.get()

    local_seed = args.real_seed + device_thread_id * args.local_num_envs
    np.random.seed(local_seed)

    envs = make_env(
        args,
        local_seed,
        args.local_num_envs,
        args.local_env_threads,
        thread_affinity_offset=device_thread_id * args.local_env_threads,
    )
    envs = RecordEpisodeStatistics(envs)

    eval_envs = make_env(
        args,
        local_seed + 100000,
        args.local_eval_episodes,
        args.local_eval_episodes // 4, mode=eval_mode, eval=True)
    eval_envs = RecordEpisodeStatistics(eval_envs)

    len_actor_device_ids = len(args.actor_device_ids)
    n_actors = args.num_actor_threads * len_actor_device_ids
    global_step = 0
    start_time = time.time()
    warmup_step = 0
    other_time = 0
    avg_ep_returns = deque(maxlen=1000)
    avg_win_rates = deque(maxlen=1000)
    avg_ep_int_rewards = deque(maxlen=1000)

    agent = create_agent(args)
    eval_agent = create_agent(args, eval=eval_mode != 'bot')

    if args.enable_rnd:
        rnd_target = create_rnd_model(args)
        rnd_predictor = create_rnd_model(args, predictor=True)

    @jax.jit
    def get_action(params, obs, rstate):
        rstate, logits = eval_agent.apply(params, obs, rstate)[:2]
        return rstate, logits.argmax(axis=1)

    @jax.jit
    def get_action_battle(params1, params2, obs, rstate1, rstate2, main, done):
        next_rstate1, logits1 = agent.apply(params1, obs, rstate1)[:2]
        next_rstate2, logits2 = eval_agent.apply(params2, obs, rstate2)[:2]
        logits = jnp.where(main[:, None], logits1, logits2)
        rstate1 = jax.tree.map(
            lambda x1, x2: jnp.where(main[:, None], x1, x2), next_rstate1, rstate1)
        rstate2 = jax.tree.map(
            lambda x1, x2: jnp.where(main[:, None], x2, x1), next_rstate2, rstate2)
        rstate1, rstate2 = jax.tree.map(
            lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
        return rstate1, rstate2, logits.argmax(axis=1)
    
    @jax.jit
    def compute_int_rew(params_rt, params_rp, obs):
        target_feats = rnd_target.apply(params_rt, obs)
        predict_feats = rnd_predictor.apply(params_rp, obs)
        int_rewards = jnp.sum((target_feats - predict_feats) ** 2, axis=-1) / 2
        if args.rnd_norm == 'min_max':
            int_rewards = (int_rewards - int_rewards.min()) / (int_rewards.max() - int_rewards.min() + 1e-8)
        return target_feats, int_rewards

    @jax.jit
    def sample_action(
        params, next_obs, rstate1, rstate2, main, done, key,
        params_rt, params_rp, rewems):
        (rstate1, rstate2), logits = agent.apply(
            params, next_obs, (rstate1, rstate2), done, main)[:2]
        action, key = categorical_sample(logits, key)

        if args.enable_rnd:
            target_feats, int_rewards = compute_int_rew(params_rt, params_rp, next_obs)
            if args.rnd_norm == 'default':
                rewems = rewems * args.int_gamma + int_rewards
        else:
            target_feats = int_rewards = None
        return next_obs, done, main, rstate1, rstate2, action, logits, key, \
            target_feats, int_rewards, rewems

    deck_names = args.deck_names
    deck_avg_times = {name: 0 for name in deck_names}
    deck_max_times = {name: 0 for name in deck_names}
    deck_time_count = {name: 0 for name in deck_names}

    # put data in the last index
    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, info = envs.reset()
    next_to_play = info["to_play"]
    next_done = np.zeros(args.local_num_envs, dtype=np.bool_)
    next_rstate1 = next_rstate2 = agent.init_rnn_state(args.local_num_envs)

    eval_rstate1 = agent.init_rnn_state(args.local_eval_episodes)
    eval_rstate2 = eval_agent.init_rnn_state(args.local_eval_episodes)

    next_rstate1, next_rstate2, eval_rstate1, eval_rstate2 = \
        jax.device_put([next_rstate1, next_rstate2, eval_rstate1, eval_rstate2], actor_device)

    main_player = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(main_player)
    storage = []

    reward_rms = RunningMeanStd()
    rewems = jnp.zeros(args.local_num_envs, dtype=jnp.float32, device=actor_device)

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree.map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    for update in range(1, args.num_updates + 2):
        if update == 10:
            start_time = time.time()
            warmup_step = global_step

        update_time_start = time.time()
        inference_time = 0
        env_time = 0
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:
                params, params_rp = params_queue.get()
                # params["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
                actor_policy_version += 1
        else:
            params, params_rp = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)

        rollout_time_start = time.time()
        init_rstate1, init_rstate2 = jax.tree.map(
            lambda x: x.copy(), (next_rstate1, next_rstate2))

        all_int_rewards = []
        all_dis_int_rewards = []
        for k in range(args.num_steps):
            global_step += args.local_num_envs * n_actors * args.world_size

            main = next_to_play == main_player

            inference_time_start = time.time()
            cached_next_obs, cached_next_done, cached_main, \
                next_rstate1, next_rstate2, action, logits, key, \
                target_feats, int_rewards, rewems = sample_action(
                params, next_obs, next_rstate1, next_rstate2,
                main, next_done, key, params_rt, params_rp, rewems)

            cpu_action = np.array(action)
            inference_time += time.time() - inference_time_start

            _start = time.time()
            next_obs, next_reward, next_done, info = envs.step(cpu_action)
            next_to_play = info["to_play"]
            env_time += time.time() - _start

            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_done,
                    mains=cached_main,
                    actions=action,
                    logits=logits,
                    rewards=next_reward * args.reward_scale,
                    next_dones=next_done,
                    target_feats=target_feats,
                )
            )
            if args.enable_rnd:
                all_int_rewards.append(int_rewards)
                all_dis_int_rewards.append(rewems)

            for idx, d in enumerate(next_done):
                if not d:
                    continue
                cur_main = main[idx]
                if args.switch:
                    for j in reversed(range(len(storage) - 1)):
                        t = storage[j]
                        if t.next_dones[idx]:
                            # For OTK where player may not switch
                            break
                        if t.mains[idx] != cur_main:
                            t.next_dones[idx] = True
                            t.rewards[idx] = -next_reward[idx]
                            break
                
                if args.time_log_freq:
                    for i in range(2):
                        deck_time = info['step_time'][idx][i]
                        deck_name = deck_names[info['deck'][idx][i]]

                        time_count = deck_time_count[deck_name]
                        avg_time = deck_avg_times[deck_name]
                        avg_time = avg_time * (time_count / (time_count + 1)) + deck_time / (time_count + 1)
                        max_time = max(deck_time, deck_max_times[deck_name])
                        deck_avg_times[deck_name] = avg_time
                        deck_max_times[deck_name] = max_time
                        deck_time_count[deck_name] += 1
                        if deck_time_count[deck_name] % args.time_log_freq == 0:
                            print(f"Deck {deck_name}, avg: {avg_time * 1000:.2f}, max: {max_time * 1000:.2f}")

                episode_reward = info['r'][idx] * (1 if cur_main else -1)
                win = 1 if episode_reward > 0 else 0
                avg_ep_returns.append(episode_reward)
                avg_win_rates.append(win)
                
                avg_ep_int_rewards.append(float(int_rewards[idx]))

        rollout_time.append(time.time() - rollout_time_start)

        if args.enable_rnd:
            next_int_reward = compute_int_rew(params_rt, params_rp, next_obs)[1]
            all_int_rewards = all_int_rewards[1:] + [next_int_reward]

            # TODO: update every step
            all_int_rewards = jnp.stack(all_int_rewards)
            if args.rnd_norm == 'default':
                all_dis_int_rewards = jnp.concatenate(all_dis_int_rewards)
                mean, std = jax.device_get((
                    all_dis_int_rewards.mean(), all_dis_int_rewards.std()))
                count = len(all_dis_int_rewards)
                reward_rms.update_from_moments(mean, std**2, count)
                all_int_rewards = all_int_rewards / np.sqrt(reward_rms.var)
            elif args.rnd_norm == 'min_max2':
                max_int_rewards = jnp.max(all_int_rewards)
                min_int_rewards = jnp.min(all_int_rewards)
                all_int_rewards = (all_int_rewards - min_int_rewards) / (max_int_rewards - min_int_rewards)
            mean_int_rewards = jnp.mean(all_int_rewards)
            max_int_rewards = jnp.max(all_int_rewards)

            for k in range(args.num_steps):
                int_rewards = all_int_rewards[k]
                storage[k] = storage[k]._replace(int_rewards=int_rewards)


        partitioned_storage = prepare_data(storage)
        storage = []
        sharded_storage = []
        for x in partitioned_storage:
            if isinstance(x, dict):
                x = {
                    k: jax.device_put_sharded(v, devices=learner_devices)
                    for k, v in x.items()
                }
            elif x is not None:
                x = jax.device_put_sharded(x, devices=learner_devices)
            sharded_storage.append(x)
        sharded_storage = Transition(*sharded_storage)
        next_main = main_player == next_to_play
        next_rstate = jax.tree.map(
            lambda x1, x2: jnp.where(next_main[:, None], x1, x2), next_rstate1, next_rstate2)
        sharded_data = jax.tree.map(lambda x: jax.device_put_sharded(
                np.split(x, len(learner_devices)), devices=learner_devices),
                         (init_rstate1, init_rstate2, (next_obs, next_rstate), next_main))

        if args.eval_interval and update % args.eval_interval == 0:
            _start = time.time()
            if eval_mode == 'bot':
                predict_fn = lambda *x: get_action(params, *x)
                eval_return, eval_ep_len, eval_win_rate = evaluate(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate2)
            else:
                predict_fn = lambda *x: get_action_battle(params, eval_params, *x)
                eval_return, eval_ep_len, eval_win_rate = battle(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate1, eval_rstate2)
            eval_time = time.time() - _start
            other_time += eval_time
            eval_stats = np.array([eval_time, eval_return, eval_win_rate], dtype=np.float32)
        else:
            eval_stats = None

        payload = (
            global_step,
            update,
            sharded_storage,
            *sharded_data,
            np.mean(params_queue_get_time),
            eval_stats,
        )
        rollout_queue.put(payload)

        if update % args.log_frequency == 0:
            avg_episodic_return = np.mean(avg_ep_returns)
            avg_episodic_length = np.mean(envs.returned_episode_lengths)
            SPS = int((global_step - warmup_step) / (time.time() - start_time - other_time))
            SPS_update = int(args.batch_size / (time.time() - update_time_start))

            tb_global_step = args.tb_offset + global_step

            if device_thread_id == 0:
                print(
                    f"global_step={tb_global_step}, avg_return={avg_episodic_return:.4f}, avg_length={avg_episodic_length:.0f}"
                )
                time_now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                print(
                    f"{time_now} SPS: {SPS}, update: {SPS_update}, "
                    f"rollout_time={rollout_time[-1]:.2f}, params_time={params_queue_get_time[-1]:.2f}"
                )
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), tb_global_step)
            if args.enable_rnd:
                writer.add_scalar("charts/avg_episodic_int_rew", np.mean(avg_ep_int_rewards), tb_global_step)
                writer.add_scalar("charts/mean_int_rew", float(mean_int_rewards), tb_global_step)
                writer.add_scalar("charts/max_int_rew", float(max_int_rewards), tb_global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, tb_global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, tb_global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), tb_global_step)
            writer.add_scalar("stats/inference_time", inference_time, tb_global_step)
            writer.add_scalar("stats/env_time", env_time, tb_global_step)
            writer.add_scalar("charts/SPS", SPS, tb_global_step)
            writer.add_scalar("charts/SPS_update", SPS_update, tb_global_step)


def main():
    args = tyro.cli(Args)
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    args.local_env_threads = args.local_env_threads or args.local_num_envs
    if args.segment_length is not None:
        assert args.num_steps % args.segment_length == 0, "num_steps must be divisible by segment_length"
    args.enable_rnd = args.int_coef > 0

    if args.embedding_file:
        embeddings = load_embeddings(args.embedding_file, args.code_list_file)
        embedding_shape = embeddings.shape
        args.num_embeddings = embedding_shape
        args.freeze_id = True if args.freeze_id is None else args.freeze_id
    else:
        embeddings = None
        embedding_shape = None

    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    global_main_devices = [
        global_devices[process_index * len(local_devices)]
        for process_index in range(args.world_size)
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    if args.run_name is None:
        timestamp = int(time.time())
        run_name = f"{args.exp_name}__{args.seed}__{timestamp}"
    else:
        run_name = args.run_name
        timestamp = int(run_name.split("__")[-1])

    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    tb_log_dir = f"{args.tb_dir}/{run_name}"
    if args.local_rank == 0 and not args.debug:
        writer = SummaryWriter(tb_log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = dummy_writer

    def save_fn(obj, path):
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(obj))

    ckpt_maneger = ModelCheckpoint(
        args.ckpt_dir, save_fn, n_saved=2)

    # seeding
    random.seed(args.seed)
    seed = random.randint(0, int(1e8))

    seed_offset = args.local_rank
    seed += seed_offset
    init_key = jax.random.PRNGKey(seed - seed_offset)

    random.seed(seed)
    args.real_seed = random.randint(0, int(1e8))

    key = jax.random.PRNGKey(args.real_seed)
    key, *learner_keys = jax.random.split(key, len(learner_devices) + 1)
    learner_keys = jax.device_put_sharded(learner_keys, devices=learner_devices)
    actor_keys = jax.random.split(key, len(actor_devices) * args.num_actor_threads)

    deck, deck_names = init_ygopro(args.env_id, "english", args.deck, args.code_list_file, return_deck_names=True)
    args.deck_names = sorted(deck_names)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, 0, 2, 1)
    obs_space = envs.observation_space
    action_shape = envs.action_space.shape
    print(f"obs_space={obs_space}, action_shape={action_shape}")
    sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    envs.close()
    del envs

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    agent = create_agent(args)
    rstate = agent.init_rnn_state(1)
    params = agent.init(init_key, sample_obs, rstate)
    if embeddings is not None:
        unknown_embed = embeddings.mean(axis=0)
        embeddings = np.concatenate([unknown_embed[None, :], embeddings], axis=0)
        params = flax.core.unfreeze(params)
        params['params']['Encoder_0']['Embed_0']['embedding'] = jax.device_put(embeddings)
        params = flax.core.freeze(params)
    
    if args.enable_rnd:
        rnd_init_key1, rnd_init_key2 = jax.random.split(init_key, 2)
        rnd_target = create_rnd_model(args)
        rnd_predictor = create_rnd_model(args, predictor=True)
        params_rt = rnd_target.init(rnd_init_key1, sample_obs)
        params_rp = rnd_predictor.init(rnd_init_key2, sample_obs)
    else:
        params_rt = params_rp = None

    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
        every_k_schedule=1,
    )
    tx = optax.apply_if_finite(tx, max_consecutive_errors=10)
    agent_state = TrainState.create(
        apply_fn=None,
        params=(params, params_rp),
        tx=tx,
    )
    # TODO: checkpoint for RND
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
            agent_state = agent_state.replace(params=(params, params_rp))
        print(f"loaded checkpoint from {args.checkpoint}")

    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    # print(agent.tabulate(agent_key, sample_obs))

    if args.eval_checkpoint:
        eval_agent = create_agent(args, eval=True)
        eval_rstate = eval_agent.init_rnn_state(1)
        eval_params = eval_agent.init(init_key, sample_obs, eval_rstate)
        with open(args.eval_checkpoint, "rb") as f:
            eval_params = flax.serialization.from_bytes(eval_params, f.read())
        print(f"loaded eval checkpoint from {args.eval_checkpoint}")
    else:
        eval_params = None

    def advantage_fn(
        new_logits, new_values, next_dones, switch_or_mains,
        actions, logits, rewards, next_value):
        num_envs = jax.tree.leaves(next_value)[0].shape[0]
        num_steps = next_dones.shape[0] // num_envs

        def reshape_time_series(x):
            return jnp.reshape(x, (num_steps, num_envs) + x.shape[1:])

        ratios = distrax.importance_sampling_ratios(distrax.Categorical(
            new_logits), distrax.Categorical(logits), actions)

        new_values_, rewards, next_dones, switch_or_mains = jax.tree.map(
            reshape_time_series, (new_values, rewards, next_dones, switch_or_mains),
        )

        # Advantages and target values
        if args.switch:
            target_values, advantages = gae_2p0s_switch(
                next_value, new_values_, rewards, next_dones, switch_or_mains,
                args.gamma, args.gae_lambda, args.upgo)
        else:
            # TODO: TD(lambda) for multi-step
            ratios_ = reshape_time_series(ratios)
            if args.enable_rnd:
                new_values_, new_values_int_ = new_values_
                next_value, next_value_int = next_value
                rewards, rewards_int = rewards
            target_values, advantages = vtrace_2p0s(
                next_value, ratios_, new_values_, rewards, next_dones, switch_or_mains, args.gamma,
                args.rho_clip_min, args.rho_clip_max, args.c_clip_min, args.c_clip_max)
            if args.enable_rnd:
                next_dones = next_dones if args.rnd_episodic else jnp.zeros_like(next_dones)
                target_values_int, advantages_int = truncated_gae(
                    next_value_int, new_values_int_, rewards_int, next_dones, args.int_gamma, args.gae_lambda)
                advantages = advantages * args.ext_coef + advantages_int * args.int_coef
                target_values = (target_values, target_values_int)

        target_values, advantages = jax.tree.map(
            lambda x: jnp.reshape(x, (-1,)), (target_values, advantages))
        return target_values, advantages

    def loss_fn(
        new_logits, new_values, actions, logits, target_values, advantages,
        mask, num_steps=None):
        ratios = distrax.importance_sampling_ratios(distrax.Categorical(
            new_logits), distrax.Categorical(logits), actions)
        logratio = jnp.log(ratios)
        approx_kl = (ratios - 1) - logratio

        if args.norm_adv:
            advantages = masked_normalize(advantages, mask, eps=1e-8)

        # Policy loss
        if args.spo_kld_max is not None:
            pg_loss = simple_policy_loss(
                ratios, logits, new_logits, advantages, args.spo_kld_max)
        elif args.logits_threshold is not None:
            pg_loss = ach_loss(
                actions, logits, new_logits, advantages, args.logits_threshold, args.clip_coef, args.dual_clip_coef)
        elif args.ppo_clip:
            pg_loss = clipped_surrogate_pg_loss(
                ratios, advantages, args.clip_coef, args.dual_clip_coef)
        else:
            pg_advs = jnp.clip(ratios, args.rho_clip_min, args.rho_clip_max) * advantages
            pg_loss = policy_gradient_loss(new_logits, actions, pg_advs)

        v_loss = 0
        if args.enable_rnd:
            new_values, new_values_int = new_values
            target_values, target_values_int = target_values
            int_v_loss = mse_loss(new_values_int, target_values_int)
            v_loss += int_v_loss

        v_loss += mse_loss(new_values, target_values)
        if args.vloss_clip is not None:
            v_loss = jnp.minimum(v_loss, args.vloss_clip)

        ent_loss = entropy_loss(new_logits)

        if args.burn_in_steps:
            mask = jax.tree.map(
                lambda x: x.reshape(num_steps, -1), mask)
            burn_in_mask = jnp.arange(num_steps) < args.burn_in_steps
            mask = jnp.where(burn_in_mask[:, None], 0.0, mask)
            mask = jnp.reshape(mask, (-1,))

        n_valids = jnp.sum(mask)
        pg_loss, v_loss, ent_loss, approx_kl = jax.tree.map(
            lambda x: jnp.sum(x * mask) / n_valids, (pg_loss, v_loss, ent_loss, approx_kl))

        loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
        return loss, pg_loss, v_loss, ent_loss, approx_kl

    def compute_rnd_loss(params, obs, mask, target_feats, key):
        n_use = int(mask.shape[0] * args.rnd_update_proportion)
        obs, mask, target_feats = jax.tree.map(
            lambda x: jax.random.permutation(key, x)[:n_use], (obs, mask, target_feats))
        predict_feats = rnd_predictor.apply(params, obs)
        rnd_loss = mse_loss(predict_feats, target_feats).mean(axis=-1)
        rnd_loss = (rnd_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        return rnd_loss

    def apply_fn(params, obs, rstate1, rstate2, dones, next_dones, switch_or_mains):
        if args.switch:
            dones = dones | next_dones
        (rstate1, rstate2), new_logits, new_values = agent.apply(
            params, obs, (rstate1, rstate2), dones, switch_or_mains)[:3]
        new_values = jax.tree.map(lambda x: x.squeeze(-1), new_values)
        return (rstate1, rstate2), new_logits, new_values

    def compute_advantage(
        params, rstate1, rstate2, obs, dones, next_dones,
        switch_or_mains, actions, logits, rewards, next_value):
        segment_length = dones.shape[0]

        obs, dones, next_dones, switch_or_mains, actions, logits, rewards = \
            jax.tree.map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                (obs, dones, next_dones, switch_or_mains, actions, logits, rewards))

        new_logits, new_values = apply_fn(
            params, obs, rstate1, rstate2, dones, next_dones, switch_or_mains)[1:3]

        target_values, advantages = advantage_fn(
            new_logits, new_values, next_dones, switch_or_mains,
            actions, logits, rewards, next_value)

        target_values, advantages = jax.tree.map(
            lambda x: jnp.reshape(x, (segment_length, -1) + x.shape[2:]),
            (target_values, advantages))
        return target_values, advantages

    def compute_loss(
        params, rstate1, rstate2, obs, dones, next_dones,
        switch_or_mains, actions, logits, target_values, advantages, mask,
        target_feats, key):
        params, params_rp = params
        (rstate1, rstate2), new_logits, new_values = apply_fn(
            params, obs, rstate1, rstate2, dones, next_dones, switch_or_mains)

        loss, pg_loss, v_loss, ent_loss, approx_kl = loss_fn(
            new_logits, new_values, actions, logits, target_values, advantages,
            mask, num_steps=None)

        if args.enable_rnd:
            rnd_loss = compute_rnd_loss(params_rp, obs, mask, target_feats, key)
            loss = loss + rnd_loss
        else:
            rnd_loss = jnp.zeros_like(loss)

        loss = jnp.where(jnp.isnan(loss) | jnp.isinf(loss), 0.0, loss)
        approx_kl, rstate1, rstate2 = jax.tree.map(
            jax.lax.stop_gradient, (approx_kl, rstate1, rstate2))
        return loss, (pg_loss, v_loss, ent_loss, rnd_loss, approx_kl, rstate1, rstate2)

    def compute_advantage_loss(
        params, rstate1, rstate2, obs, dones, next_dones,
        switch_or_mains, actions, logits, rewards, next_value, mask,
        target_feats, key):
        num_envs = jax.tree.leaves(next_value)[0].shape[0]
        params, params_rp = params
        new_logits, new_values = apply_fn(
            params, obs, rstate1, rstate2, dones, next_dones, switch_or_mains)[1:3]

        target_values, advantages = advantage_fn(
            new_logits, new_values, next_dones, switch_or_mains,
            actions, logits, rewards, next_value)

        loss, pg_loss, v_loss, ent_loss, approx_kl = loss_fn(
            new_logits, new_values, actions, logits, target_values, advantages,
            mask, num_steps=dones.shape[0] // num_envs)

        if args.enable_rnd:
            rnd_loss = compute_rnd_loss(params_rp, obs, mask, target_feats, key)
            loss = loss + rnd_loss
        else:
            rnd_loss = jnp.zeros_like(loss)

        loss = jnp.where(jnp.isnan(loss) | jnp.isinf(loss), 0.0, loss)
        approx_kl = jax.lax.stop_gradient(approx_kl)
        return loss, (pg_loss, v_loss, ent_loss, rnd_loss, approx_kl)

    def split_key_if(key, n, cond):
        if cond:
            return jax.random.split(key, n)
        else:
            return [key] * n

    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List,
        sharded_init_rstate1: List,
        sharded_init_rstate2: List,
        sharded_next_inputs: List,
        sharded_next_main: List,
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
        # TODO: rstate will be out-date after the first update, maybe consider R2D2
        next_inputs, init_rstate1, init_rstate2 = [
            jax.tree.map(lambda *x: jnp.concatenate(x), *x)
            for x in [sharded_next_inputs, sharded_init_rstate1, sharded_init_rstate2]
        ]
        next_main = jnp.concatenate(sharded_next_main)

        # reorder storage of individual players
        # main first, opponent second
        num_steps, num_envs = storage.rewards.shape
        if args.switch:
            T = jnp.arange(num_steps, dtype=jnp.int32)
            B = jnp.arange(num_envs, dtype=jnp.int32)
            mains = storage.mains.astype(jnp.int32)
            indices = jnp.argsort(T[:, None] - mains * num_steps, axis=0)
            switch_steps = jnp.sum(mains, axis=0)
            switch = T[:, None] == (switch_steps[None, :] - 1)
            storage = jax.tree.map(lambda x: x[indices, B[None, :]], storage)

        if args.segment_length is None:
            loss_grad_fn = jax.value_and_grad(compute_advantage_loss, has_aux=True)
        else:
            loss_grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            next_value = agent.apply(agent_state.params[0], *next_inputs)[2]
            next_value = jax.tree.map(lambda x: jnp.squeeze(x, axis=-1), next_value)

            if args.enable_rnd:
                next_value, next_value_int = next_value
            sign = -1 if args.switch else 1
            next_value = jnp.where(next_main, sign * next_value, -sign * next_value)
            if args.enable_rnd:
                next_value = next_value, next_value_int

            def convert_data(x: jnp.ndarray, multi_step=True):
                key = subkey if args.update_epochs > 1 else None
                return reshape_minibatch(
                    x, multi_step, args.num_minibatches, num_steps, args.segment_length, key=key)

            shuffled_init_rstate1, shuffled_init_rstate2 = jax.tree.map(
                partial(convert_data, multi_step=False), (init_rstate1, init_rstate2))
            shuffled_storage = jax.tree.map(convert_data, storage)
            if args.switch:
                switch_or_mains = convert_data(switch)
            else:
                switch_or_mains = shuffled_storage.mains
            shuffled_mask = ~shuffled_storage.dones
            shuffled_next_value = jax.tree.map(
                partial(convert_data, multi_step=False), next_value)
            shuffled_rewards = shuffled_storage.rewards
            if args.enable_rnd:
                shuffled_rewards = shuffled_rewards, shuffled_storage.int_rewards

            if args.segment_length is None:
                def update_minibatch(carry, minibatch):
                    agent_state, key = carry
                    key, subkey = split_key_if(key, 2, args.enable_rnd)
                    (loss, (pg_loss, v_loss, ent_loss, rnd_loss, approx_kl)), grads = loss_grad_fn(
                        agent_state.params, *minibatch, subkey)
                    grads = jax.lax.pmean(grads, axis_name="local_devices")
                    agent_state = agent_state.apply_gradients(grads=grads)
                    return (agent_state, key), (loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl)
            else:
                def update_minibatch(carry, minibatch):
                    def update_minibatch_t(carry, minibatch_t):
                        (agent_state, key), rstate1, rstate2 = carry
                        key, subkey = split_key_if(key, 2, args.enable_rnd)
                        minibatch_t = rstate1, rstate2, *minibatch_t
                        (loss, (pg_loss, v_loss, ent_loss, approx_kl, rstate1, rstate2)), \
                            grads = loss_grad_fn(agent_state.params, *minibatch_t, subkey)
                        grads = jax.lax.pmean(grads, axis_name="local_devices")
                        agent_state = agent_state.apply_gradients(grads=grads)
                        return ((agent_state, key), rstate1, rstate2), (loss, pg_loss, v_loss, ent_loss, approx_kl)

                    rstate1, rstate2, *minibatch_t, mask = minibatch
                    target_values, advantages = compute_advantage(
                        carry[0].params[0], rstate1, rstate2, *minibatch_t)
                    minibatch_t = *minibatch_t[:-2], target_values, advantages, mask

                    (carry, _rstate1, _rstate2), \
                        (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(
                        update_minibatch_t, (carry, rstate1, rstate2), minibatch_t)
                    return carry, (loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl)

            (agent_state, key), (loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl) = jax.lax.scan(
                update_minibatch,
                (agent_state, key),
                (
                    shuffled_init_rstate1,
                    shuffled_init_rstate2,
                    shuffled_storage.obs,
                    shuffled_storage.dones,
                    shuffled_storage.next_dones,
                    switch_or_mains,
                    shuffled_storage.actions,
                    shuffled_storage.logits,
                    shuffled_rewards,
                    shuffled_next_value,
                    shuffled_mask,
                    shuffled_storage.target_feats,
                ),
            )
            return (agent_state, key), (loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl)

        (agent_state, key), (loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        ent_loss = jax.lax.pmean(ent_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        rnd_loss = jax.lax.pmean(
            rnd_loss, axis_name="local_devices").mean() if args.enable_rnd else 0
        return agent_state, loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl, key

    all_reduce_value = jax.pmap(
        lambda x: jax.lax.pmean(x, axis_name="main_devices"),
        axis_name="main_devices",
        devices=global_main_devices,
    )
    
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    params_queues = []
    rollout_queues = []

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        actor_device = local_devices[d_id]
        device_params = jax.device_put(unreplicated_params, actor_device)
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            init_params = [params_rt]
            if eval_params:
                init_params.append(eval_params)
            params_queues[-1].put(
                jax.device_put(init_params, actor_device))
            actor_thread_id = d_idx * args.num_actor_threads + thread_id             
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(actor_keys[actor_thread_id], actor_device),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    actor_device,
                    learner_devices,
                    actor_thread_id,
                ),
            ).start()
            params_queues[-1].put(device_params)

    rollout_queue_get_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_data_list = []
        eval_stat_list = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    update,
                    *sharded_data,
                    avg_params_queue_get_time,
                    eval_stats,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_data_list.append(sharded_data)
                if eval_stats is not None:
                    eval_stat_list.append(eval_stats)

        tb_global_step = args.tb_offset + global_step
        if update % args.eval_interval == 0:
            eval_stats = np.mean(eval_stat_list, axis=0)
            eval_stats = jax.device_put(eval_stats, local_devices[0])
            eval_stats = np.array(all_reduce_value(eval_stats[None])[0])
            eval_time, eval_return, eval_win_rate = eval_stats
            writer.add_scalar(f"charts/eval_return", eval_return, tb_global_step)
            writer.add_scalar(f"charts/eval_win_rate", eval_win_rate, tb_global_step)
            print(f"eval_time={eval_time:.4f}, eval_return={eval_return:.4f}, eval_win_rate={eval_win_rate:.4f}")

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, ent_loss, rnd_loss, approx_kl, learner_keys) = multi_device_update(
            agent_state,
            *list(zip(*sharded_data_list)),
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        params_queue_put_time = 0
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            device_params[0]["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
            params_queue_put_start = time.time()
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)
            params_queue_put_time += time.time() - params_queue_put_start

        loss = loss[-1].item()
        if np.isnan(loss) or np.isinf(loss):
            raise ValueError(f"loss is {loss}")

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), tb_global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                tb_global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, tb_global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), tb_global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), tb_global_step)
            print(
                f"{tb_global_step} actor_update={update}, "
                f"train_time={time.time() - training_time_start:.2f}, "
                f"data_time={rollout_queue_get_time[-1]:.2f}, "
                f"put_time={params_queue_put_time:.2f}"
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[3][2][1].hyperparams["learning_rate"][-1].item(), tb_global_step
            )
            if args.enable_rnd:
                writer.add_scalar("losses/rnd_loss", rnd_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/entropy", ent_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1].item(), tb_global_step)
            writer.add_scalar("losses/loss", loss, tb_global_step)

        if args.local_rank == 0 and learner_policy_version % args.save_interval == 0 and not args.debug:
            M_steps = tb_global_step // 2**20
            ckpt_name = f"{timestamp}_{M_steps}M.flax_model"
            ckpt_maneger.save(unreplicated_params[0], ckpt_name)
            if args.gcs_bucket is not None:
                lastest_path = ckpt_maneger.get_latest()
                copy_path = lastest_path.with_name("latest" + lastest_path.suffix)
                shutil.copyfile(lastest_path, copy_path)
                zip_file_path = "latest.zip"
                zip_files(zip_file_path, [str(copy_path), tb_log_dir])
                sync_to_gcs(args.gcs_bucket, zip_file_path)

        if learner_policy_version >= args.num_updates:
            break

    if args.distributed:
        jax.distributed.shutdown()

    writer.close()


if __name__ == "__main__":
    main()
