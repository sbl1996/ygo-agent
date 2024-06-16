from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

import chex
import distrax


def entropy_loss(logits):
    return distrax.Softmax(logits=logits).entropy()


def mse_loss(y_true, y_pred):
    return 0.5 * ((y_true - y_pred) ** 2)


def policy_gradient_loss(logits, actions, advantages):
    chex.assert_type([logits, actions, advantages], [float, int, float])
  
    advs = jax.lax.stop_gradient(advantages)
    log_probs = distrax.Softmax(logits=logits).log_prob(actions)
    pg_loss = -log_probs * advs
    return pg_loss


def clipped_surrogate_pg_loss(ratios, advantages, clip_coef, dual_clip_coef=None):
    # dual clip from JueWu (Mastering Complex Control in MOBA Games with Deep Reinforcement Learning)
    advs = jax.lax.stop_gradient(advantages)
    clipped_ratios = jnp.clip(ratios, 1 - clip_coef, 1 + clip_coef)
    clipped_obj = jnp.fmin(ratios * advs, clipped_ratios * advs)
    if dual_clip_coef is not None:
        clipped_obj = jnp.where(
            advs >= 0, clipped_obj,
            jnp.fmax(clipped_obj, dual_clip_coef * advs)
        )
    pg_loss = -clipped_obj
    return pg_loss


def get_from_action(values, action):
    num_categories = values.shape[-1]
    value_one_hot = jax.nn.one_hot(
        action, num_categories, dtype=values.dtype)
    return jnp.sum(distrax.multiply_no_nan(values, value_one_hot), axis=-1)


def mean_legal(values, axis=None, keepdims=False):
    # TODO: use real action mask
    no_nan_mask = values > -1e12
    no_nan = jnp.where(no_nan_mask, values, 0)
    count = jnp.sum(no_nan_mask, axis=axis, keepdims=keepdims)
    return jnp.sum(no_nan, axis=axis, keepdims=keepdims) / jnp.maximum(count, 1)


def neurd_loss_2(actions, logits, new_logits, advantages, logits_threshold):
    # Neural Replicator Dynamics
    # Differences from the original implementation:
    # - all actions vs. sampled actions
    # - original computes advantages with q values
    # - original does not use importance sampling ratios
    advs = jax.lax.stop_gradient(advantages)
    probs_a = get_from_action(jax.nn.softmax(logits), actions)
    probs_a = jnp.maximum(probs_a, 0.001)
    new_logits_a = get_from_action(new_logits, actions)
    new_logits_a_ = new_logits_a - mean_legal(new_logits, axis=-1)

    can_decrease_1 = new_logits_a_ < logits_threshold
    can_decrease_2 = new_logits_a_ > -logits_threshold
    c = jnp.where(
        advs >= 0, can_decrease_1, can_decrease_2).astype(jnp.float32)
    c = jax.lax.stop_gradient(c)
    pg_loss = -c * new_logits_a_ / probs_a * advs
    return pg_loss


def neurd_loss(new_logits, advantages, logits_threshold=2.0, adv_threshold=1000.0):
    advs = jax.lax.stop_gradient(advantages)

    legal_mask = new_logits > -1e12
    legal_logits = jnp.where(legal_mask, new_logits, 0)
    count = jnp.sum(legal_mask, axis=-1, keepdims=True)
    new_logits_ = new_logits - jnp.sum(legal_logits, axis=-1, keepdims=True) / jnp.maximum(count, 1)

    can_increase = new_logits_ < logits_threshold
    can_decrease = new_logits_ > -logits_threshold
    c = jnp.where(
        advs >= 0, can_increase, can_decrease).astype(jnp.float32)
    c = jax.lax.stop_gradient(c)
    advs = jnp.clip(advs, -adv_threshold, adv_threshold)
    # TODO: renormalize with player
    pg_loss = -c * new_logits_ * advs
    pg_loss = jnp.where(legal_mask, pg_loss, 0)
    pg_loss = jnp.sum(pg_loss, axis=-1)
    return pg_loss


def ach_loss(actions, logits, new_logits, advantages, logits_threshold, clip_coef, dual_clip_coef=None):
    # Actor-Critic Hedge loss from Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game
    # notice entropy term is required but not included here
    advs = jax.lax.stop_gradient(advantages)
    probs_a = get_from_action(jax.nn.softmax(logits), actions)
    probs_a = jnp.maximum(probs_a, 0.001)
    new_logits_a = get_from_action(new_logits, actions)
    new_logits_a_ = new_logits_a - mean_legal(new_logits, axis=-1)

    ratios = distrax.importance_sampling_ratios(distrax.Categorical(
        new_logits), distrax.Categorical(logits), actions)
    can_decrease_1 = (ratios < 1 + clip_coef) * (new_logits_a_ < logits_threshold)
    can_decrease_2 = (ratios > 1 - clip_coef) * (new_logits_a_ > -logits_threshold)
    if dual_clip_coef is not None:
        can_decrease_2 = can_decrease_2 * (ratios < dual_clip_coef)
    c = jnp.where(
        advs >= 0, can_decrease_1, can_decrease_2).astype(jnp.float32)
    c = jax.lax.stop_gradient(c)
    pg_loss = -c * new_logits_a_ / probs_a * advs
    return pg_loss


def vtrace_rnad_loop(carry, inp, gamma, rho_min, rho_max, c_min, c_max):
    v1, v2, next_values1, next_values2, reward1, reward2, xi1, xi2, reward_u1, reward_u2 = carry
    ratio, cur_values, r_t, eta_reg_entropy, probs, a_t, eta_log_policy, next_done, main = inp

    v1 = jnp.where(next_done, 0, v1)
    v2 = jnp.where(next_done, 0, v2)
    next_values1 = jnp.where(next_done, 0, next_values1)
    next_values2 = jnp.where(next_done, 0, next_values2)
    reward1 = jnp.where(next_done, 0, reward1)
    reward2 = jnp.where(next_done, 0, reward2)
    xi1 = jnp.where(next_done, 1, xi1)
    xi2 = jnp.where(next_done, 1, xi2)
    reward_u1 = jnp.where(next_done, 0, reward_u1)
    reward_u2 = jnp.where(next_done, 0, reward_u2)

    discount = gamma * (1.0 - next_done)
    next_v = jnp.where(main, v1, v2)
    next_values = jnp.where(main, next_values1, next_values2)
    reward = jnp.where(main, reward1, reward2)
    xi = jnp.where(main, xi1, xi2)
    reward_u = jnp.where(main, reward_u1, reward_u2)
    
    reward_u = r_t + discount * reward_u + eta_reg_entropy
    discounted_reward = r_t + discount * reward

    rho_t = jnp.clip(ratio * xi, rho_min, rho_max)
    c_t = jnp.clip(ratio * xi, c_min, c_max)
    sig_v = rho_t * (reward_u + discount * next_values - cur_values)
    v = cur_values + sig_v + c_t * discount * (next_v - next_values)

    q_t = cur_values[:, None] + eta_log_policy
    n_actions = eta_log_policy.shape[-1]
    q_t2 = discounted_reward + discount * xi * next_v - cur_values
    q_t = q_t + q_t2[:, None] * distrax.multiply_no_nan(
        1.0 / jnp.maximum(probs, 1e-3), jax.nn.one_hot(a_t, n_actions))

    v1 = jnp.where(main, v, discount * v1)
    v2 = jnp.where(main, discount * v2, v)
    next_values1 = jnp.where(main, cur_values, discount * next_values1)
    next_values2 = jnp.where(main, discount * next_values2, cur_values)
    reward1 = jnp.where(main, 0, ratio * (discount * reward1 - r_t) - eta_reg_entropy)
    reward2 = jnp.where(main, ratio * (discount * reward2 - r_t) - eta_reg_entropy, 0)
    xi1 = jnp.where(main, 1, ratio * xi1)
    xi2 = jnp.where(main, ratio * xi2, 1)
    reward_u1 = jnp.where(main, 0, discount * reward_u1 - r_t - eta_reg_entropy)
    reward_u2 = jnp.where(main, discount * reward_u2 - r_t - eta_reg_entropy, 0)

    carry = v1, v2, next_values1, next_values2, reward1, reward2, xi1, xi2, reward_u1, reward_u2
    return carry, (v, q_t)


def vtrace_rnad(
    next_value, ratios, logits, new_logits, actions,
    log_policy_reg, values, rewards, next_dones, mains,
    gamma, rho_min=0.001, rho_max=1.0, c_min=0.001, c_max=1.0, eta=0.2,
):
    probs = jax.nn.softmax(logits)
    new_probs = jax.nn.softmax(new_logits)
    eta_reg_entropy = -eta * jnp.sum(new_probs * log_policy_reg, axis=-1)
    eta_log_policy = -eta * log_policy_reg
    next_value1 = next_value
    next_value2 = -next_value1
    v1 = next_value1
    v2 = next_value2
    reward1 = reward2 = reward_u1 = reward_u2 = jnp.zeros_like(next_value)
    xi1 = xi2 = jnp.ones_like(next_value)
    carry = v1, v2, next_value1, next_value2, reward1, reward2, xi1, xi2, reward_u1, reward_u2

    _, (targets, q_estimate) = jax.lax.scan(
        partial(vtrace_rnad_loop, gamma=gamma, rho_min=rho_min, rho_max=rho_max, c_min=c_min, c_max=c_max),
        carry, (ratios, values, rewards, eta_reg_entropy, probs, actions, eta_log_policy, next_dones, mains), reverse=True
    )
    targets = jax.lax.stop_gradient(targets)
    return targets, q_estimate


class VtraceCarry(NamedTuple):
    v: jnp.ndarray
    next_value: jnp.ndarray
    last_return: jnp.ndarray
    next_q: jnp.ndarray
    next_main: jnp.ndarray


def vtrace_loop(carry, inp, gamma, rho_min, rho_max, c_min, c_max):
    v, next_value, last_return, next_q, next_main = carry
    ratio, cur_value, next_done, reward, main = inp

    v = jnp.where(next_done, 0, v)
    next_value = jnp.where(next_done, 0, next_value)
    
    sign = jnp.where(main == next_main, 1, -1)
    v = v * sign
    next_value = next_value * sign

    discount = gamma * (1.0 - next_done)

    q_t = reward + discount * v

    rho_t = jnp.clip(ratio, rho_min, rho_max)
    c_t = jnp.clip(ratio, c_min, c_max)
    sig_v = rho_t * (reward + discount * next_value - cur_value)
    v = cur_value + sig_v + c_t * discount * (v - next_value)

    # UPGO advantage (not corrected by importance sampling, unlike V-trace)
    last_return = last_return * sign
    next_q = next_q * sign
    last_return = reward + discount * jnp.where(
        next_q >= next_value, last_return, next_value)

    next_q = reward + discount * next_value

    carry = VtraceCarry(v, next_value, last_return, next_q, main)
    return carry, (v, q_t, last_return)


def vtrace(
    next_v, ratios, values, rewards, next_dones, mains,
    gamma, rho_min=0.001, rho_max=1.0, c_min=0.001, c_max=1.0,
    upgo=False, return_carry=False
):
    if isinstance(next_v, (tuple, list)):
        carry = next_v
    else:
        next_value = next_v
        v = last_return = next_q = next_value
        next_main = jnp.ones_like(next_value, dtype=jnp.bool_)
        carry = VtraceCarry(v, next_value, last_return, next_q, next_main)

    carry, (targets, q_estimate, return_t) = jax.lax.scan(
        partial(vtrace_loop, gamma=gamma, rho_min=rho_min, rho_max=rho_max, c_min=c_min, c_max=c_max),
        carry, (ratios, values, next_dones, rewards, mains), reverse=True
    )
    if return_carry:
        return carry
    advantages = q_estimate - values
    if upgo:
        advantages += return_t - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages


class VtraceSepCarry(NamedTuple):
    v: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    xi: jnp.ndarray
    last_return: jnp.ndarray
    next_q: jnp.ndarray


def vtrace_sep_loop(carry, inp, gamma, rho_min, rho_max, c_min, c_max):
    (v1, next_value1, reward1, xi1, last_return1, next_q1), \
        (v2, next_value2, reward2, xi2, last_return2, next_q2) = carry
    ratio, cur_value, next_done, r_t, main = inp


    v1, v2, next_value1, next_value2, reward1, reward2, xi1, xi2 = jax.tree.map(
        lambda x: jnp.where(next_done, 0, x),
        (v1, v2, next_value1, next_value2, reward1, reward2, xi1, xi2))

    discount = gamma * (1.0 - next_done)
    v = jnp.where(main, v1, v2)
    next_value = jnp.where(main, next_value1, next_value2)
    reward = jnp.where(main, reward1, reward2)
    xi = jnp.where(main, xi1, xi2)

    q_t = r_t + ratio * reward + discount * v

    rho_t = jnp.clip(ratio * xi, rho_min, rho_max)
    c_t = jnp.clip(ratio * xi, c_min, c_max)
    sig_v = rho_t * (r_t + ratio * reward + discount * next_value - cur_value)
    v = cur_value + sig_v + c_t * discount * (v - next_value)

    # UPGO advantage (not corrected by importance sampling, unlike V-trace)
    return_t = jnp.where(main, last_return1, last_return2)
    next_q = jnp.where(main, next_q1, next_q2)
    factor = jnp.where(main, jnp.ones_like(r_t), -jnp.ones_like(r_t))
    return_t = r_t + discount * jnp.where(
        next_q >= next_value, return_t, next_value)
    last_return1 = jnp.where(
        next_done, r_t * factor, jnp.where(main, return_t, last_return1))
    last_return2 = jnp.where(
        next_done, r_t * -factor, jnp.where(main, last_return2, return_t))
    next_q = r_t + discount * next_value
    next_q1 = jnp.where(
        next_done, r_t * factor, jnp.where(main, next_q, next_q1))
    next_q2 = jnp.where(
        next_done, r_t * -factor, jnp.where(main, next_q2, next_q))

    v1 = jnp.where(main, v, v1)
    v2 = jnp.where(main, v2, v)
    next_value1 = jnp.where(main, cur_value, next_value1)
    next_value2 = jnp.where(main, next_value2, cur_value)
    reward1 = jnp.where(main, 0, -r_t + ratio * reward1)
    reward2 = jnp.where(main, -r_t + ratio * reward2, 0)
    xi1 = jnp.where(main, 1, ratio * xi1)
    xi2 = jnp.where(main, ratio * xi2, 1)

    carry1 = VtraceSepCarry(v1, next_value1, reward1, xi1, last_return1, next_q1)
    carry2 = VtraceSepCarry(v2, next_value2, reward2, xi2, last_return2, next_q2)
    return (carry1, carry2), (v, q_t, return_t)


def vtrace_sep(
    next_v, ratios, values, rewards, next_dones, mains,
    gamma, rho_min=0.001, rho_max=1.0, c_min=0.001, c_max=1.0,
    upgo=False, return_carry=False
):
    if isinstance(next_v, (tuple, list)):
        carry = next_v
    else:
        next_value = next_v
        next_value1 = next_value
        carry1 = VtraceSepCarry(
            v=next_value1,
            next_value=next_value1,
            reward=jnp.zeros_like(next_value),
            xi=jnp.ones_like(next_value),
            last_return=next_value1,
            next_q=next_value1,
        )
        next_value2 = -next_value1
        carry2 = VtraceSepCarry(
            v=next_value2,
            next_value=next_value2,
            reward=jnp.zeros_like(next_value),
            xi=jnp.ones_like(next_value),
            last_return=next_value2,
            next_q=next_value2,
        )
        carry = carry1, carry2

    carry, (targets, q_estimate, return_t) = jax.lax.scan(
        partial(vtrace_sep_loop, gamma=gamma, rho_min=rho_min, rho_max=rho_max, c_min=c_min, c_max=c_max),
        carry, (ratios, values, next_dones, rewards, mains), reverse=True
    )
    if return_carry:
        return carry
    advantages = q_estimate - values
    if upgo:
        advantages += return_t - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages


class GAESepCarry(NamedTuple):
    lastgaelam: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    done_used: jnp.ndarray
    last_return: jnp.ndarray
    next_q: jnp.ndarray


def truncated_gae_sep_loop(carry, inp, gamma, gae_lambda):
    (lastgaelam1, next_value1, reward1, done_used1, last_return1, next_q1), \
         (lastgaelam2, next_value2, reward2, done_used2, last_return2, next_q2) = carry
    cur_value, next_done, reward, main = inp
    main1 = main
    main2 = ~main
    factor = jnp.where(main1, jnp.ones_like(reward), -jnp.ones_like(reward))
    reward1 = jnp.where(next_done, reward * factor, jnp.where(main1 & done_used1, 0, reward1))
    reward2 = jnp.where(next_done, reward * -factor, jnp.where(main2 & done_used2, 0, reward2))
    real_done1 = next_done | ~done_used1
    next_value1 = jnp.where(real_done1, 0, next_value1)
    lastgaelam1 = jnp.where(real_done1, 0, lastgaelam1)
    real_done2 = next_done | ~done_used2
    next_value2 = jnp.where(real_done2, 0, next_value2)
    lastgaelam2 = jnp.where(real_done2, 0, lastgaelam2)
    done_used1 = jnp.where(
        next_done, main1, jnp.where(main1 & ~done_used1, True, done_used1))
    done_used2 = jnp.where(
        next_done, main2, jnp.where(main2 & ~done_used2, True, done_used2))

    last_return1 = jnp.where(real_done1, 0, last_return1)
    last_return2 = jnp.where(real_done2, 0, last_return2)
    last_return1_ = reward1 + gamma * jnp.where(
        next_q1 >= next_value1, last_return1, next_value1)
    last_return2_ = reward2 + gamma * jnp.where(
        next_q2 >= next_value2, last_return2, next_value2)
    return_t = jnp.where(main1, last_return1_, last_return2_)
    last_return1 = jnp.where(main1, last_return1_, last_return1)
    last_return2 = jnp.where(main2, last_return2_, last_return2)
    next_q1_ = reward1 + gamma * next_value1
    next_q2_ = reward2 + gamma * next_value2
    next_q1 = jnp.where(main1, next_q1_, next_q1)
    next_q2 = jnp.where(main2, next_q2_, next_q2)

    delta1 = reward1 + gamma * next_value1 - cur_value
    delta2 = reward2 + gamma * next_value2 - cur_value
    lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
    lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
    advantages = jnp.where(main1, lastgaelam1_, lastgaelam2_)
    next_value1 = jnp.where(main1, cur_value, next_value1)
    next_value2 = jnp.where(main2, cur_value, next_value2)
    lastgaelam1 = jnp.where(main1, lastgaelam1_, lastgaelam1)
    lastgaelam2 = jnp.where(main2, lastgaelam2_, lastgaelam2)

    carry1 = GAESepCarry(lastgaelam1, next_value1, reward1, done_used1, last_return1, next_q1)
    carry2 = GAESepCarry(lastgaelam2, next_value2, reward2, done_used2, last_return2, next_q2)
    return (carry1, carry2), (advantages, return_t)


def truncated_gae_sep(
    next_v, values, rewards, next_dones, mains, gamma, gae_lambda, upgo, return_carry=False):
    if isinstance(next_v, (tuple, list)):
        carry = next_v
    else:
        next_value = next_v
        carry1 = GAESepCarry(
            lastgaelam=jnp.zeros_like(next_value),
            next_value=next_value,
            reward=jnp.zeros_like(next_value),
            done_used=jnp.ones_like(next_dones[-1]),
            last_return=next_value,
            next_q=next_value,
        )
        carry2 = GAESepCarry(
            lastgaelam=jnp.zeros_like(next_value),
            next_value=-next_value,
            reward=jnp.zeros_like(next_value),
            done_used=jnp.ones_like(next_dones[-1]),
            last_return=-next_value,
            next_q=-next_value,
        )
        carry = carry1, carry2
    carry, (advantages, returns) = jax.lax.scan(
        partial(truncated_gae_sep_loop, gamma=gamma, gae_lambda=gae_lambda),
        carry, (values, next_dones, rewards, mains), reverse=True
    )
    if return_carry:
        return carry
    targets = values + advantages
    if upgo:
        advantages += returns - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages


class GAECarry(NamedTuple):
    lastgaelam: jnp.ndarray
    next_value: jnp.ndarray
    last_return: jnp.ndarray
    next_q: jnp.ndarray
    next_main: jnp.ndarray


def truncated_gae_loop(carry, inp, gamma, gae_lambda):
    lastgaelam, next_value, last_return, next_q, next_main = carry
    cur_value, next_done, reward, main = inp
    
    lastgaelam = jnp.where(next_done, 0, lastgaelam)
    next_value = jnp.where(next_done, 0, next_value)

    sign = jnp.where(main == next_main, 1, -1)
    lastgaelam = lastgaelam * sign
    next_value = next_value * sign

    discount = gamma * (1.0 - next_done)

    delta = reward + discount * next_value - cur_value
    lastgaelam = delta + discount * gae_lambda * lastgaelam

    # UPGO advantage
    last_return = last_return * sign
    next_q = next_q * sign
    last_return = reward + discount * jnp.where(
        next_q >= next_value, last_return, next_value)

    next_q = reward + discount * next_value

    carry = GAECarry(lastgaelam, cur_value, last_return, next_q, main)
    return carry, (lastgaelam, last_return)


def truncated_gae(
    next_v, values, rewards, next_dones, mains, gamma, gae_lambda,
    upgo=False, return_carry=False):
    if isinstance(next_v, (tuple, list)):
        carry = next_v
    else:
        next_value = next_v
        carry = GAECarry(
            lastgaelam=jnp.zeros_like(next_value),
            next_value=next_value,
            last_return=next_value,
            next_q=next_value,
            next_main=jnp.ones_like(next_value, dtype=jnp.bool_),
        )
    carry, (advantages, return_t) = jax.lax.scan(
        partial(truncated_gae_loop, gamma=gamma, gae_lambda=gae_lambda),
        carry, (values, next_dones, rewards, mains), reverse=True
    )
    if return_carry:
        return carry
    targets = values + advantages
    if upgo:
        advantages += return_t - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages


def simple_policy_loss(ratios, logits, new_logits, advantages, kld_max, eps=1e-12):
    advs = jax.lax.stop_gradient(advantages)
    probs = jax.nn.softmax(logits)
    new_probs = jax.nn.softmax(new_logits)
    kld = jnp.sum(
        probs * jnp.log((probs + eps) / (new_probs + eps)), axis=-1)
    kld_clip = jnp.clip(kld, 0, kld_max)
    d_ratio = kld_clip / (kld + eps)

    # e == 1 and t == 1
    d_ratio = jnp.where(kld < 1e-6, 1.0, d_ratio)

    sign_a = jnp.sign(advs)
    result = (d_ratio + sign_a - 1) * sign_a
    pg_loss = -advs * ratios * result
    return pg_loss