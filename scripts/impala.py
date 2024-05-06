import os
import shutil
import queue
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, NamedTuple, Optional
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
from ygoai.rl.jax.agent2 import PPOLSTMAgent
from ygoai.rl.jax.utils import RecordEpisodeStatistics, categorical_sample
from ygoai.rl.jax.eval import evaluate, battle
from ygoai.rl.jax import vtrace_2p0s, clipped_surrogate_pg_loss, policy_gradient_loss, mse_loss, entropy_loss


os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    log_frequency: int = 10
    """the logging frequency of the model performance (in terms of `updates`)"""
    save_interval: int = 400
    """the frequency of saving the model (in terms of `updates`)"""
    checkpoint: Optional[str] = None
    """the path to the model checkpoint to load"""
    debug: bool = False
    """whether to run the script in debug mode"""

    tb_dir: str = "runs"
    """the directory to save the tensorboard logs"""
    ckpt_dir: str = "checkpoints"
    """the directory to save the model checkpoints"""
    gcs_bucket: Optional[str] = None
    """the GCS bucket to save the model checkpoints"""

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
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    upgo: bool = True
    """Toggle the use of UPGO for advantages"""
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
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    rnn_channels: int = 512
    """the number of channels for the RNN in the agent"""

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


def create_agent(args, multi_step=False):
    return PPOLSTMAgent(
        channels=args.num_channels,
        num_layers=args.num_layers,
        embedding_shape=args.num_embeddings,
        dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
        param_dtype=jnp.float32,
        lstm_channels=args.rnn_channels,
        switch=False,
        multi_step=multi_step,
        freeze_id=args.freeze_id,
    )


def init_rnn_state(num_envs, rnn_channels):
    return (
        np.zeros((num_envs, rnn_channels)),
        np.zeros((num_envs, rnn_channels)),
    )


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue,
    writer,
    learner_devices,
    device_thread_id,
):
    eval_mode = 'self' if args.eval_checkpoint else 'bot'
    if eval_mode != 'bot':
        eval_params = params_queue.get()

    local_seed = args.seed + device_thread_id
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
        local_seed,
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

    @jax.jit
    def get_logits(
        params: flax.core.FrozenDict, inputs):
        rstate, logits = create_agent(args).apply(params, inputs)[:2]
        return rstate, logits

    @jax.jit
    def get_action(
        params: flax.core.FrozenDict, inputs):
        rstate, logits = get_logits(params, inputs)
        return rstate, logits.argmax(axis=1)

    @jax.jit
    def get_action_battle(params1, params2, rstate1, rstate2, obs, main, done):
        next_rstate1, logits1 = get_logits(params1, (rstate1, obs))
        next_rstate2, logits2 = get_logits(params2, (rstate2, obs))
        logits = jnp.where(main[:, None], logits1, logits2)
        rstate1 = jax.tree.map(
            lambda x1, x2: jnp.where(main[:, None], x1, x2), next_rstate1, rstate1)
        rstate2 = jax.tree.map(
            lambda x1, x2: jnp.where(main[:, None], x2, x1), next_rstate2, rstate2)
        rstate1, rstate2 = jax.tree.map(
            lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
        return rstate1, rstate2, logits.argmax(axis=1)

    @jax.jit
    def sample_action(
        params: flax.core.FrozenDict,
        next_obs, rstate1, rstate2, main, done, key):
        next_obs = jax.tree.map(lambda x: jnp.array(x), next_obs)
        done = jnp.array(done)
        main = jnp.array(main)

        rstate = jax.tree.map(
            lambda x1, x2: jnp.where(main[:, None], x1, x2), rstate1, rstate2)
        rstate, logits = get_logits(params, (rstate, next_obs))
        rstate1 = jax.tree.map(lambda x, y: jnp.where(main[:, None], x, y), rstate, rstate1)
        rstate2 = jax.tree.map(lambda x, y: jnp.where(main[:, None], y, x), rstate, rstate2)
        rstate1, rstate2 = jax.tree.map(
            lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
        action, key = categorical_sample(logits, key)
        return next_obs, done, main, rstate1, rstate2, action, logits, key

    # put data in the last index
    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, info = envs.reset()
    next_to_play = info["to_play"]
    next_done = np.zeros(args.local_num_envs, dtype=np.bool_)
    next_rstate1 = next_rstate2 = init_rnn_state(
        args.local_num_envs, args.rnn_channels)
    eval_rstate = init_rnn_state(
        args.local_eval_episodes, args.rnn_channels)
    main_player = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(main_player)
    storage = []

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
                params = params_queue.get()
                # params["params"]["Encoder_0"]['Embed_0'][
                #     "embedding"
                # ].block_until_ready()
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)

        rollout_time_start = time.time()
        init_rstate1, init_rstate2 = jax.tree.map(
            lambda x: x.copy(), (next_rstate1, next_rstate2))
        for _ in range(args.num_steps):
            global_step += args.local_num_envs * n_actors * args.world_size

            cached_next_obs = next_obs
            cached_next_done = next_done
            main = next_to_play == main_player

            inference_time_start = time.time()
            cached_next_obs, cached_next_done, cached_main, \
                next_rstate1, next_rstate2, action, logits, key = sample_action(
                params, cached_next_obs, next_rstate1, next_rstate2, main, cached_next_done, key)
            
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
                    rewards=next_reward,
                    next_dones=next_done,
                )
            )

            for idx, d in enumerate(next_done):
                if not d:
                    continue
                cur_main = main[idx]
                episode_reward = info['r'][idx] * (1 if cur_main else -1)
                win = 1 if episode_reward > 0 else 0
                avg_ep_returns.append(episode_reward)
                avg_win_rates.append(win)

        rollout_time.append(time.time() - rollout_time_start)

        partitioned_storage = prepare_data(storage)
        storage = []
        sharded_storage = []
        for x in partitioned_storage:
            if isinstance(x, dict):
                x = {
                    k: jax.device_put_sharded(v, devices=learner_devices)
                    for k, v in x.items()
                }
            else:
                x = jax.device_put_sharded(x, devices=learner_devices)
            sharded_storage.append(x)
        sharded_storage = Transition(*sharded_storage)
        next_main = main_player == next_to_play
        next_rstate = jax.tree.map(
            lambda x1, x2: jnp.where(next_main[:, None], x1, x2), next_rstate1, next_rstate2)
        sharded_data = jax.tree.map(lambda x: jax.device_put_sharded(
                np.split(x, len(learner_devices)), devices=learner_devices),
                         (init_rstate1, init_rstate2, (next_rstate, next_obs), next_main))

        if args.eval_interval and update % args.eval_interval == 0:
            _start = time.time()
            if eval_mode == 'bot':
                predict_fn = lambda x: get_action(params, x)
                eval_return, eval_ep_len, eval_win_rate = evaluate(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate)
            else:
                predict_fn = lambda *x: get_action_battle(params, eval_params, *x)
                eval_return, eval_ep_len, eval_win_rate = battle(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate)
            eval_time = time.time() - _start
            other_time += eval_time
            eval_stats = np.array([eval_time, eval_return, eval_win_rate], dtype=np.float32)
        else:
            eval_stats = None

        learn_opponent = False
        payload = (
            global_step,
            update,
            sharded_storage,
            *sharded_data,
            np.mean(params_queue_get_time),
            learn_opponent,
            eval_stats,
        )
        rollout_queue.put(payload)

        if update % args.log_frequency == 0:
            avg_episodic_return = np.mean(avg_ep_returns)
            avg_episodic_length = np.mean(envs.returned_episode_lengths)
            SPS = int((global_step - warmup_step) / (time.time() - start_time - other_time))
            SPS_update = int(args.batch_size / (time.time() - update_time_start))
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_return={avg_episodic_return:.4f}, avg_length={avg_episodic_length:.0f}"
                )
                time_now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                print(
                    f"{time_now} SPS: {SPS}, update: {SPS_update}, "
                    f"rollout_time={rollout_time[-1]:.2f}, params_time={params_queue_get_time[-1]:.2f}"
                )
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/env_time", env_time, global_step)
            writer.add_scalar("charts/SPS", SPS, global_step)
            writer.add_scalar("charts/SPS_update", SPS_update, global_step)


if __name__ == "__main__":
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

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

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
    seed_offset = args.local_rank * 10000
    args.seed += seed_offset
    random.seed(args.seed)
    init_key = jax.random.PRNGKey(args.seed - seed_offset)
    key = jax.random.PRNGKey(args.seed)
    key, *learner_keys = jax.random.split(key, len(learner_devices) + 1)
    learner_keys = jax.device_put_sharded(learner_keys, devices=learner_devices)
    actor_keys = jax.random.split(key, len(actor_devices) * args.num_actor_threads)

    deck = init_ygopro(args.env_id, "english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, args.seed, 8, 1)
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

    rstate = init_rnn_state(1, args.rnn_channels)
    agent = create_agent(args)
    params = agent.init(init_key, (rstate, sample_obs))
    if embeddings is not None:
        unknown_embed = embeddings.mean(axis=0)
        embeddings = np.concatenate([unknown_embed[None, :], embeddings], axis=0)
        params = flax.core.unfreeze(params)
        params['params']['Encoder_0']['Embed_0']['embedding'] = jax.device_put(embeddings)
        params = flax.core.freeze(params)

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
        params=params,
        tx=tx,
    )
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
            agent_state = agent_state.replace(params=params)
        print(f"loaded checkpoint from {args.checkpoint}")

    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    # print(agent.tabulate(agent_key, sample_obs))

    if args.eval_checkpoint:
        with open(args.eval_checkpoint, "rb") as f:
            eval_params = flax.serialization.from_bytes(params, f.read())
        print(f"loaded eval checkpoint from {args.eval_checkpoint}")
    else:
        eval_params = None

    @jax.jit
    def get_logits_and_value(
        params: flax.core.FrozenDict, inputs,
    ):
        rstate, logits, value, valid = create_agent(
            args, multi_step=True).apply(params, inputs)
        return logits, value.squeeze(-1)

    def loss_fn(
        params, rstate1, rstate2, obs, dones, next_dones,
        mains, actions, logits, rewards, mask, next_value):
        # (num_steps * local_num_envs // n_mb))
        num_envs = next_value.shape[0]
        num_steps = dones.shape[0] // num_envs

        def reshape_time_series(x):
            return jnp.reshape(x, (num_steps, num_envs) + x.shape[1:])

        mask = mask * (1.0 - dones)
        n_valids = jnp.sum(mask)

        inputs = (rstate1, rstate2, obs, dones, mains)
        new_logits, new_values = get_logits_and_value(params, inputs)

        ratios = distrax.importance_sampling_ratios(distrax.Categorical(
            new_logits), distrax.Categorical(logits), actions)
        logratio = jnp.log(ratios)
        approx_kl = (((ratios - 1) - logratio) * mask).sum() / n_valids

        ratios_, new_values_, rewards, next_dones, mains = jax.tree.map(
            reshape_time_series, (ratios, new_values, rewards, next_dones, mains),
        )

        # TODO: TD(lambda) for multi-step
        target_values, advantages = vtrace_2p0s(
            next_value, ratios_, new_values_, rewards, next_dones, mains, args.gamma,
            args.rho_clip_min, args.rho_clip_max, args.c_clip_min, args.c_clip_max)
        target_values, advantages = jax.tree.map(
            lambda x: jnp.reshape(x, (-1,)), (target_values, advantages))

        if args.ppo_clip:
            pg_loss = clipped_surrogate_pg_loss(
                ratios, advantages, args.clip_coef, args.dual_clip_coef)
        else:
            pg_advs = jnp.clip(ratios, args.rho_clip_min, args.rho_clip_max) * advantages
            pg_loss = policy_gradient_loss(new_logits, actions, pg_advs)
        pg_loss = jnp.sum(pg_loss * mask)

        v_loss = mse_loss(new_values, target_values)
        v_loss = jnp.sum(v_loss * mask)

        ent_loss = entropy_loss(new_logits)
        ent_loss = jnp.sum(ent_loss * mask)

        pg_loss = pg_loss / n_valids
        v_loss = v_loss / n_valids
        ent_loss = ent_loss / n_valids

        loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, ent_loss, jax.lax.stop_gradient(approx_kl))

    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List,
        sharded_init_rstate1: List,
        sharded_init_rstate2: List,
        sharded_next_inputs: List,
        sharded_next_main: List,
        key: jax.random.PRNGKey,
        learn_opponent: bool = False,
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

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            next_value = create_agent(args).apply(
                agent_state.params, next_inputs)[2].squeeze(-1)
            next_value = jnp.where(next_main, next_value, -next_value)

            def convert_data(x: jnp.ndarray, num_steps):
                if args.update_epochs > 1:
                    x = jax.random.permutation(subkey, x, axis=1 if num_steps > 1 else 0)
                N = args.num_minibatches
                if num_steps > 1:
                    x = jnp.reshape(x, (num_steps, N, -1) + x.shape[2:])
                    x = x.transpose(1, 0, *range(2, x.ndim))
                    x = x.reshape(N, -1, *x.shape[3:])
                else:
                    x = jnp.reshape(x, (N, -1) + x.shape[1:])
                return x

            shuffled_init_rstate1, shuffled_init_rstate2, \
                shuffled_next_value = jax.tree.map(
                partial(convert_data, num_steps=1), (init_rstate1, init_rstate2, next_value))
            shuffled_storage = jax.tree.map(
                partial(convert_data, num_steps=num_steps), storage)
            shuffled_mask = jnp.ones_like(shuffled_storage.mains)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = loss_grad_fn(
                    agent_state.params, *minibatch)
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    shuffled_init_rstate1,
                    shuffled_init_rstate2,
                    shuffled_storage.obs,
                    shuffled_storage.dones,
                    shuffled_storage.next_dones,
                    shuffled_storage.mains,
                    shuffled_storage.actions,
                    shuffled_storage.logits,
                    shuffled_storage.rewards,
                    shuffled_mask,
                    shuffled_next_value,
                ),
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    all_reduce_value = jax.pmap(
        lambda x: jax.lax.pmean(x, axis_name="main_devices"),
        axis_name="main_devices",
        devices=global_main_devices,
    )
    
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
        static_broadcasted_argnums=(7,),
    )

    params_queues = []
    rollout_queues = []

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            if eval_params:
                params_queues[-1].put(
                    jax.device_put(eval_params, local_devices[d_id]))
            actor_thread_id = d_idx * args.num_actor_threads + thread_id             
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(actor_keys[actor_thread_id], local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    actor_thread_id,
                ),
            ).start()
            params_queues[-1].put(device_params)

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
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
                    learn_opponent,
                    eval_stats,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_data_list.append(sharded_data)
                if eval_stats is not None:
                    eval_stat_list.append(eval_stats)

        if update % args.eval_interval == 0:
            eval_stats = np.mean(eval_stat_list, axis=0)
            eval_stats = jax.device_put(eval_stats, local_devices[0])
            eval_stats = np.array(all_reduce_value(eval_stats[None])[0])
            eval_time, eval_return, eval_win_rate = eval_stats
            writer.add_scalar(f"charts/eval_return", eval_return, global_step)
            writer.add_scalar(f"charts/eval_win_rate", eval_win_rate, global_step)
            print(f"eval_time={eval_time:.4f}, eval_return={eval_return:.4f}, eval_win_rate={eval_win_rate:.4f}")

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, learner_keys) = multi_device_update(
            agent_state,
            *list(zip(*sharded_data_list)),
            learner_keys,
            learn_opponent,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            device_params["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        loss = loss[-1].item()
        if np.isnan(loss) or np.isinf(loss):
            raise ValueError(f"loss is {loss}")

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
            print(
                f"{global_step} actor_update={update}, "
                f"train_time={time.time() - training_time_start:.2f}, "
                f"data_time={rollout_queue_get_time[-1]:.2f}"
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[3][2][1].hyperparams["learning_rate"][-1].item(), global_step
            )
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1].item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1].item(), global_step)
            writer.add_scalar("losses/loss", loss, global_step)

        if args.local_rank == 0 and learner_policy_version % args.save_interval == 0 and not args.debug:
            M_steps = args.batch_size * learner_policy_version // 2**20
            ckpt_name = f"{timestamp}_{M_steps}M.flax_model"
            ckpt_maneger.save(unreplicated_params, ckpt_name)
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