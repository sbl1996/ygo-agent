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

import ygoenv
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent2 import PPOAgent
from ygoai.rl.jax.utils import RecordEpisodeStatistics, masked_mean, masked_normalize
from ygoai.rl.jax.eval import evaluate
from ygoai.rl.jax import compute_gae, compute_gae_upgo


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
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    local_num_envs: int = 128
    """the number of parallel game environments"""
    local_env_threads: Optional[int] = None
    """the number of threads to use for environment"""
    num_actor_threads: int = 2
    """the number of actor threads to use"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    collect_length: Optional[int] = None
    """the number of steps to compute the advantages"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    upgo: bool = False
    """Toggle the use of UPGO for advantages"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.25
    """the surrogate clipping coefficient"""
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
    logprobs: list
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
    def get_logits(
        params: flax.core.FrozenDict,
        next_obs,
    ):
        return create_agent(args).apply(params, next_obs)[0]

    def get_action(
        params: flax.core.FrozenDict,
        next_obs,
    ):
        return get_logits(params, next_obs).argmax(axis=1)

    @jax.jit
    def sample_action(
        params: flax.core.FrozenDict,
        next_obs,
        key: jax.random.PRNGKey,
    ):
        next_obs = jax.tree_map(lambda x: jnp.array(x), next_obs)
        logits = get_logits(params, next_obs)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return next_obs, action, logprob, key

    # put data in the last index
    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, info = envs.reset()
    next_to_play = info["to_play"]
    next_done = np.zeros(args.local_num_envs, dtype=np.bool_)
    ai_player1 = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(ai_player1)
    start_step = 0
    storage = []

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
        for _ in range(start_step, args.collect_length):
            global_step += args.local_num_envs * n_actors * args.world_size

            cached_next_obs = next_obs
            cached_next_done = next_done
            learn = next_to_play == ai_player1

            inference_time_start = time.time()
            cached_next_obs, action, logprob, key = sample_action(params, cached_next_obs, key)
            cpu_action = np.array(action)
            inference_time += time.time() - inference_time_start

            _start = time.time()
            to_play = next_to_play
            next_obs, next_reward, next_done, info = envs.step(cpu_action)
            next_to_play = info["to_play"]
            env_time += time.time() - _start

            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_done,
                    actions=action,
                    logprobs=logprob,
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

        start_step = args.collect_length - args.num_steps

        partitioned_storage = prepare_data(storage)
        storage = storage[args.num_steps:]
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
        next_learn = ai_player1 == next_to_play
        sharded_data = jax.tree_map(lambda x: jax.device_put_sharded(
                np.split(x, len(learner_devices)), devices=learner_devices),
                         (next_obs, next_done, next_learn))
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            *sharded_data,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue.put(payload)

        if update % args.log_frequency == 0:
            avg_episodic_return = np.mean(avg_ep_returns)
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
    args.collect_length = args.collect_length or args.num_steps
    assert args.collect_length >= args.num_steps, "collect_length must be greater than or equal to num_steps"

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
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
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
        every_k_schedule=1,
    )
    agent_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )

    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    # print(agent.tabulate(agent_key, sample_obs))

    @jax.jit
    def get_logprob_entropy_value(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
        actions: np.ndarray,
    ):
        logits, value, valid = create_agent(args).apply(params, obs)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        return logprob, entropy, value.squeeze(), valid

    def ppo_loss(params, obs, actions, logprobs, advantages, target_values):
        newlogprob, entropy, newvalue, valid = get_logprob_entropy_value(params, obs, actions)
        logratio = newlogprob - logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        
        if args.norm_adv:
            advantages = masked_normalize(advantages, valid, eps=1e-8)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)
        pg_loss = masked_mean(pg_loss, valid)

        # Value loss
        v_loss = 0.5 * ((newvalue - target_values) ** 2)
        v_loss = masked_mean(v_loss, valid)

        entropy_loss = masked_mean(entropy, valid)
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List,
        sharded_next_obs: List,
        sharded_next_done: List,
        sharded_next_learn: List,
        key: jax.random.PRNGKey,
    ):
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        next_obs = jax.tree_map(lambda *x: jnp.concatenate(x), *sharded_next_obs)
        next_done, next_learn = [
            jnp.concatenate(x) for x in [sharded_next_done, sharded_next_learn]
        ]
        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def get_value_minibatch(agent_state, mb_obs):
                values = create_agent(args).apply(agent_state.params, mb_obs)[1].squeeze(-1)
                return agent_state, values

            flatten_obs = jax.tree_map(lambda x: x.reshape((-1, args.local_minibatch_size * 8) + x.shape[2:]), storage.obs)
            _, values = jax.lax.scan(
                get_value_minibatch, agent_state, flatten_obs)
            values = values.reshape(storage.rewards.shape)

            next_value = create_agent(args).apply(agent_state.params, next_obs)[1].squeeze(-1)

            compute_gae_fn = compute_gae_upgo if args.upgo else compute_gae
            advantages, target_values = compute_gae_fn(
                next_value, next_done, next_learn,
                values, storage.rewards, storage.dones, storage.learns,
                args.gamma, args.gae_lambda)
            advantages = advantages[:args.num_steps]
            target_values = target_values[:args.num_steps]

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (-1, args.local_minibatch_size) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, jax.tree_map(lambda x: x[:args.num_steps], storage))
            flatten_advantages = flatten(advantages)
            flatten_target_values = flatten(target_values)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)
            shuffled_advantages = convert_data(flatten_advantages)
            shuffled_target_values = convert_data(flatten_target_values)

            def update_minibatch(agent_state, minibatch):
                mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_target_values = minibatch
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_advantages,
                    mb_target_values,
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    shuffled_storage.obs,
                    shuffled_storage.actions,
                    shuffled_storage.logprobs,
                    shuffled_advantages,
                    shuffled_target_values,
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
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
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
        sharded_next_obss = []
        sharded_next_dones = []
        sharded_next_learns = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    sharded_next_learn,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)
                sharded_next_learns.append(sharded_next_learn)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, learner_keys) = multi_device_update(
            agent_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_dones,
            sharded_next_learns,
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            device_params["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

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
                global_step,
                f"actor_update={update}, train_time={time.time() - training_time_start:.2f}",
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(), global_step
            )
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1].item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1].item(), global_step)

        if args.local_rank == 0 and learner_policy_version % args.save_interval == 0:
            ckpt_dir = f"checkpoints/{run_name}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model_path = ckpt_dir + "/agent.flax_model"
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