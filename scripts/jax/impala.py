import os
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
import rlax
import tyro
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent2 import PPOAgent
from ygoai.rl.jax.utils import RecordEpisodeStatistics, masked_mean, masked_normalize
from ygoai.rl.jax.eval import evaluate
from ygoai.rl.jax import vtrace, upgo_return, clipped_surrogate_pg_loss

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    log_frequency: int = 10
    """the logging frequency of the model performance (in terms of `updates`)"""
    save_interval: int = 100
    """the frequency of saving the model"""

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

    total_timesteps: int = 5000000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 64
    """the number of parallel game environments"""
    local_env_threads: Optional[int] = None
    """the number of threads to use for environment"""
    num_actor_threads: int = 2
    """the number of actor threads to use"""
    num_steps: int = 20
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    gradient_accumulation_steps: int = 1
    """the number of gradient accumulation steps before performing an optimization step"""
    c_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    c_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""
    rho_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    rho_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""
    upgo: bool = False
    """whether to use UPGO for policy update"""
    ppo_clip: bool = True
    """whether to use the PPO clipping to replace V-Trace surrogate clipping"""
    clip_coef: float = 0.25
    """the PPO surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    """the device ids that actor workers will use"""
    learner_device_ids: List[int] = field(default_factory=lambda: [1])
    """the device ids that learner workers will use"""
    distributed: bool = False
    """whether to use `jax.distirbuted`"""
    concurrency: bool = True
    """whether to run the actor and learner concurrently"""
    bfloat16: bool = True
    """whether to use bfloat16 for the agent"""
    thread_affinity: bool = False
    """whether to use thread affinity for the environment"""

    local_eval_episodes: int = 32
    """the number of episodes to evaluate the model"""
    eval_interval: int = 50
    """the number of iterations to evaluate the model"""

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
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


def make_env(args, seed, num_envs, num_threads, mode='self', thread_affinity_offset=-1):
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
        play_mode=mode,
    )
    envs.num_envs = num_envs
    return envs


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logitss: list
    rewards: list
    learns: list


def create_agent(args):
    return PPOAgent(
        channels=args.num_channels,
        num_layers=args.num_layers,
        embedding_shape=args.num_embeddings,
        dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
        param_dtype=jnp.float32,
    )


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue: queue.Queue,
    stats_queue,
    writer,
    learner_devices,
    device_thread_id,
):
    envs = make_env(
        args,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
        args.local_env_threads,
        thread_affinity_offset=device_thread_id * args.local_env_threads,
    )
    envs = RecordEpisodeStatistics(envs)

    eval_envs = make_env(
        args,
        args.seed + jax.process_index() + device_thread_id,
        args.local_eval_episodes,
        args.local_eval_episodes // 4, mode='bot')
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
    def apply_fn(
        params: flax.core.FrozenDict,
        next_obs,
    ):
        logits, value, _valid = create_agent(args).apply(params, next_obs)
        return logits, value

    def get_action(
        params: flax.core.FrozenDict,
        next_obs,
    ):
        return apply_fn(params, next_obs)[0].argmax(axis=1)

    @jax.jit
    def sample_action(
        params: flax.core.FrozenDict,
        next_obs,
        key: jax.random.PRNGKey,
    ):
        next_obs = jax.tree_map(lambda x: jnp.array(x), next_obs)
        logits = apply_fn(params, next_obs)[0]
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return next_obs, action, logits, key

    # put data in the last index
    envs.async_reset()

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    actor_policy_version = 0
    storage = []
    ai_player1 = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(ai_player1)
    next_to_play = None
    learn = np.ones(args.local_num_envs, dtype=np.bool_)

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    for update in range(1, args.num_updates + 2):
        if update == 10:
            start_time = time.time()
            warmup_step = global_step

        update_time_start = time.time()
        inference_time = 0
        env_time = 0
        num_steps_with_bootstrap = (
            args.num_steps + int(len(storage) == 0)
        )  # num_steps + 1 to get the states for value bootstrapping.
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
        for _ in range(0, num_steps_with_bootstrap):
            global_step += args.local_num_envs * n_actors * args.world_size

            _start = time.time()
            next_obs, next_reward, next_done, info = envs.recv()
            next_reward = np.where(learn, next_reward, -next_reward)
            env_time += time.time() - _start
            to_play = next_to_play
            next_to_play = info["to_play"]
            learn = next_to_play == ai_player1

            inference_time_start = time.time()
            next_obs, action, logits, key = sample_action(params, next_obs, key)
            cpu_action = np.array(action)
            inference_time += time.time() - inference_time_start

            envs.send(cpu_action)

            storage.append(
                Transition(
                    obs=next_obs,
                    dones=next_done,
                    actions=action,
                    logitss=logits,
                    rewards=next_reward,
                    learns=learn,
                )
            )

            for idx, d in enumerate(next_done):
                if not d:
                    continue
                pl = 1 if to_play[idx] == ai_player1[idx] else -1
                episode_reward = info['r'][idx] * pl
                win = 1 if episode_reward > 0 else 0
                avg_ep_returns.append(episode_reward)
                avg_win_rates.append(win)

        rollout_time.append(time.time() - rollout_time_start)

        partitioned_storage = prepare_data(storage)
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
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue.put(payload)

        # move bootstrapping step to the beginning of the next update
        storage = storage[-1:]

        if update % args.log_frequency == 0:
            avg_episodic_return = np.mean(avg_ep_returns) if len(avg_ep_returns) > 0 else 0
            avg_episodic_length = np.mean(envs.returned_episode_lengths)
            SPS = int((global_step - warmup_step) / (time.time() - start_time - other_time))
            SPS_update = int(args.batch_size / (time.time() - update_time_start))
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_return={avg_episodic_return:.4f}, avg_length={avg_episodic_length:.0f}, rollout_time={rollout_time[-1]:.2f}"
                )
                time_now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                print(f"{time_now} SPS: {SPS}, update: {SPS_update}")
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/env_time", env_time, global_step)
            writer.add_scalar("charts/SPS", SPS, global_step)
            writer.add_scalar("charts/SPS_update", SPS_update, global_step)

        if args.eval_interval and update % args.eval_interval == 0:
            # Eval with rule-based policy
            _start = time.time()
            eval_return = evaluate(eval_envs, get_action, params)[0]
            if device_thread_id != 0:
                stats_queue.put(eval_return)
            else:
                eval_stats = []
                eval_stats.append(eval_return)
                for _ in range(1, n_actors):
                    eval_stats.append(stats_queue.get())
                eval_stats = np.mean(eval_stats)
                writer.add_scalar("charts/eval_return", eval_stats, global_step)
                if device_thread_id == 0:
                    eval_time = time.time() - _start
                    print(f"eval_time={eval_time:.4f}, eval_ep_return={eval_stats:.4f}")
                    other_time += eval_time


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.local_batch_size = int(
        args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(
        args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) *
        args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(
                len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * \
        args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (
        args.local_batch_size * args.world_size)
    args.local_env_threads = args.local_env_threads or args.local_num_envs

    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [
        str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, agent_key = jax.random.split(key, 2)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    deck = init_ygopro(args.env_id, "english", args.deck, args.code_list_file)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, args.seed, 8, 1)
    obs_space = envs.observation_space
    action_shape = envs.action_space.shape
    print(f"obs_space={obs_space}, action_shape={action_shape}")
    sample_obs = jax.tree_map(lambda x: jnp.array([np.zeros((args.local_num_envs,) + x.shape[1:])]), obs_space.sample())
    envs.close()
    del envs

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches)) / args.num_updates
        return args.learning_rate * frac

    agent = create_agent(args)
    params = agent.init(agent_key, sample_obs)
    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
        every_k_schedule=args.gradient_accumulation_steps,
    )
    agent_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )

    agent_state = flax.jax_utils.replicate(
        agent_state, devices=learner_devices)
    # print(agent.tabulate(agent_key, sample_obs))

    @jax.jit
    def get_logits_and_value(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
    ):
        logits, value, valid = create_agent(args).apply(params, obs)
        return logits, value.squeeze(-1), valid

    def impala_loss(params, obs, actions, logitss, rewards, dones, learns):
        # (num_steps + 1, local_num_envs // n_mb))
        discounts = (1.0 - dones) * args.gamma
        policy_logits, newvalue, valid = jax.vmap(
            get_logits_and_value, in_axes=(None, 0))(params, obs)
        
        newvalue = jnp.where(learns, newvalue, -newvalue)
        
        v_t = newvalue[1:]
        # Remove bootstrap timestep from non-timesteps.
        v_tm1 = newvalue[:-1]
        policy_logits = policy_logits[:-1]
        logitss = logitss[:-1]
        actions = actions[:-1]
        mask = 1.0 - dones
        rewards = rewards[1:]
        discounts = discounts[1:]
        mask = mask[:-1]

        rhos = rlax.categorical_importance_sampling_ratios(
            policy_logits, logitss, actions)

        vtrace_fn = partial(
            vtrace, c_clip_min=args.c_clip_min, c_clip_max=args.c_clip_max, rho_clip_min=args.rho_clip_min, rho_clip_max=args.rho_clip_max)
        vtrace_returns = jax.vmap(
            vtrace_fn, in_axes=1, out_axes=1)(
            v_tm1, v_t, rewards, discounts, rhos)
        jax.debug.print("R {}", jnp.where(dones[1:-1, :2], rewards[:-1, :2], 0).T)
        jax.debug.print("E {}", jnp.where(dones[1:-1, :2], vtrace_returns.errors[:-1, :2] * 100, vtrace_returns.errors[:-1, :2]).T)
        jax.debug.print("V {}", v_tm1[:-1, :2].T)
        
        T = v_tm1.shape[0]
        if args.upgo:
            advs = jax.vmap(upgo_return, in_axes=1, out_axes=1)(
                rewards, v_t, discounts) - v_tm1
        else:
            advs = vtrace_returns.q_estimate - v_tm1
        if args.ppo_clip:
            pg_loss = jax.vmap(
                partial(clipped_surrogate_pg_loss, epsilon=args.clip_coef), in_axes=1)(
                rhos, advs, mask) * T
            pg_loss = jnp.sum(pg_loss)
        else:
            pg_advs = jnp.minimum(args.rho_clip_max, rhos) * advs
            pg_loss = jax.vmap(
                rlax.policy_gradient_loss, in_axes=1)(
                policy_logits, actions, pg_advs, mask) * T
            pg_loss = jnp.sum(pg_loss)

        baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)

        ent_loss = jax.vmap(rlax.entropy_loss, in_axes=1)(
            policy_logits, mask) * T
        ent_loss = jnp.sum(ent_loss)

        n_samples = jnp.sum(mask)
        pg_loss = pg_loss / n_samples
        baseline_loss = baseline_loss / n_samples
        ent_loss = ent_loss / n_samples

        total_loss = pg_loss
        total_loss += args.vf_coef * baseline_loss
        total_loss += args.ent_coef * ent_loss
        return total_loss, (pg_loss, baseline_loss, ent_loss)

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List[Transition],
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        impala_loss_grad_fn = jax.value_and_grad(impala_loss, has_aux=True)

        def update_minibatch(agent_state, minibatch):
            mb_obs, mb_actions, mb_logitss, mb_rewards, mb_dones, mb_learns = minibatch
            (loss, (pg_loss, v_loss, entropy_loss)), grads = impala_loss_grad_fn(
                agent_state.params,
                mb_obs,
                mb_actions,
                mb_logitss,
                mb_rewards,
                mb_dones,
                mb_learns,
            )
            grads = jax.lax.pmean(grads, axis_name="local_devices")
            agent_state = agent_state.apply_gradients(grads=grads)
            return agent_state, (loss, pg_loss, v_loss, entropy_loss)

        n_mb = args.num_minibatches * args.gradient_accumulation_steps
        storage_obs = {
            k: jnp.array(jnp.split(v, n_mb, axis=1))
            for k, v in storage.obs.items()
        }
        agent_state, (loss, pg_loss, v_loss, entropy_loss) = jax.lax.scan(
            update_minibatch,
            agent_state,
            (
                # (num_steps + 1, local_num_envs) => (n_mb, num_steps + 1, local_num_envs // n_mb)
                storage_obs,
                jnp.array(jnp.split(storage.actions, n_mb, axis=1)),
                jnp.array(jnp.split(storage.logitss, n_mb, axis=1)),
                jnp.array(jnp.split(storage.rewards, n_mb, axis=1)),
                jnp.array(jnp.split(storage.dones, n_mb, axis=1)),
                jnp.array(jnp.split(storage.learns, n_mb, axis=1)),
            ),
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(
            entropy_loss, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, entropy_loss, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    params_queues = []
    rollout_queues = []
    stats_queues = queue.Queue()
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(
            unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    stats_queues,
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
        rollout_queue_get_time.append(
            time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, learner_keys) = multi_device_update(
            agent_state,
            sharded_storages,
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(
                unreplicated_params, local_devices[d_id])
            device_params["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads +
                              thread_id].put(device_params)

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time",
                              np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time",
                              time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size",
                              rollout_queues[-1].qsize(), global_step)
            writer.add_scalar("stats/params_queue_size",
                              params_queues[-1].qsize(), global_step)
            print(
                global_step,
                f"actor_update={update}, train_time={time.time() - training_time_start:.2f}",
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(), global_step
            )
            writer.add_scalar("losses/value_loss",
                              v_loss[-1].item(), global_step)
            writer.add_scalar("losses/policy_loss",
                              pg_loss[-1].item(), global_step)
            writer.add_scalar("losses/entropy",
                              entropy_loss[-1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1].item(), global_step)

        if args.local_rank == 0 and learner_policy_version % args.save_interval == 0:
            ckpt_dir = f"checkpoints/{run_name}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model_path = ckpt_dir + "/agent.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            vars(args),
                            unreplicated_params,
                        ]
                    )
                )
            print(f"model saved to {model_path}")

        if learner_policy_version >= args.num_updates:
            break

    if args.distributed:
        jax.distributed.shutdown()

    writer.close()
