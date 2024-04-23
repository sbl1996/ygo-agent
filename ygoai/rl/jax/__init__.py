from functools import partial

import jax
import jax.numpy as jnp

import chex
import distrax


# class VTraceOutput(NamedTuple):
#     q_estimate: jnp.ndarray
#     errors: jnp.ndarray


# def vtrace(
#     v_tm1,
#     v_t,
#     r_t,
#     discount_t,
#     rho_tm1,
#     lambda_=1.0,
#     c_clip_min: float = 0.001,
#     c_clip_max: float = 1.007,
#     rho_clip_min: float = 0.001,
#     rho_clip_max: float = 1.007,
#     stop_target_gradients: bool = True,
# ):
#     """

#     Args:
#       v_tm1: values at time t-1.
#       v_t: values at time t.
#       r_t: reward at time t.
#       discount_t: discount at time t.
#       rho_tm1: importance sampling ratios at time t-1.
#       lambda_: mixing parameter; a scalar or a vector for timesteps t.
#       clip_rho_threshold: clip threshold for importance weights.
#       stop_target_gradients: whether or not to apply stop gradient to targets.
#     """
#     # Clip importance sampling ratios.
#     lambda_ = jnp.ones_like(discount_t) * lambda_

#     c_tm1 = jnp.clip(rho_tm1, c_clip_min, c_clip_max) * lambda_
#     clipped_rhos_tm1 = jnp.clip(rho_tm1, rho_clip_min, rho_clip_max)

#     # Compute the temporal difference errors.
#     td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

#     # Work backwards computing the td-errors.
#     def _body(acc, xs):
#         td_error, discount, c = xs
#         acc = td_error + discount * c * acc
#         return acc, acc

#     _, errors = jax.lax.scan(
#         _body, 0.0, (td_errors, discount_t, c_tm1), reverse=True)

#     # Return errors, maybe disabling gradient flow through bootstrap targets.
#     errors = jax.lax.select(
#         stop_target_gradients,
#         jax.lax.stop_gradient(errors + v_tm1) - v_tm1,
#         errors)
#     targets_tm1 = errors + v_tm1
#     q_bootstrap = jnp.concatenate([
#         lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
#         v_t[-1:],
#     ], axis=0)
#     q_estimate = r_t + discount_t * q_bootstrap
#     return VTraceOutput(q_estimate=q_estimate, errors=errors)


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


def mean_legal(values, axis=None):
    # TODO: use real action mask
    no_nan_mask = values > -1e12
    no_nan = jnp.where(no_nan_mask, values, 0)
    count = jnp.sum(no_nan_mask, axis=axis)
    return jnp.sum(no_nan, axis=axis) / jnp.maximum(count, 1)


def neurd_loss(actions, logits, new_logits, advantages, logits_threshold):
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


def vtrace_loop(carry, inp, gamma, rho_min, rho_max, c_min, c_max):
    v1, v2, next_values1, next_values2, reward1, reward2, xi1, xi2, \
        last_return1, last_return2, next_q1, next_q2 = carry
    ratio, cur_values, next_done, r_t, main = inp

    v1 = jnp.where(next_done, 0, v1)
    v2 = jnp.where(next_done, 0, v2)
    next_values1 = jnp.where(next_done, 0, next_values1)
    next_values2 = jnp.where(next_done, 0, next_values2)
    reward1 = jnp.where(next_done, 0, reward1)
    reward2 = jnp.where(next_done, 0, reward2)
    xi1 = jnp.where(next_done, 1, xi1)
    xi2 = jnp.where(next_done, 1, xi2)

    discount = gamma * (1.0 - next_done)
    v = jnp.where(main, v1, v2)
    next_values = jnp.where(main, next_values1, next_values2)
    reward = jnp.where(main, reward1, reward2)
    xi = jnp.where(main, xi1, xi2)

    q_t = r_t + ratio * reward + discount * v

    rho_t = jnp.clip(ratio * xi, rho_min, rho_max)
    c_t = jnp.clip(ratio * xi, c_min, c_max)
    sig_v = rho_t * (r_t + ratio * reward + discount * next_values - cur_values)
    v = cur_values + sig_v + c_t * discount * (v - next_values)

    # UPGO advantage (not corrected by importance sampling, unlike V-trace)
    return_t = jnp.where(main, last_return1, last_return2)
    next_q = jnp.where(main, next_q1, next_q2)
    factor = jnp.where(main, jnp.ones_like(r_t), -jnp.ones_like(r_t))
    return_t = r_t + discount * jnp.where(
        next_q >= next_values, return_t, next_values)
    last_return1 = jnp.where(
        next_done, r_t * factor, jnp.where(main, return_t, last_return1))
    last_return2 = jnp.where(
        next_done, r_t * -factor, jnp.where(main, last_return2, return_t))
    next_q = r_t + discount * next_values
    next_q1 = jnp.where(
        next_done, r_t * factor, jnp.where(main, next_q, next_q1))
    next_q2 = jnp.where(
        next_done, r_t * -factor, jnp.where(main, next_q2, next_q))

    v1 = jnp.where(main, v, v1)
    v2 = jnp.where(main, v2, v)
    next_values1 = jnp.where(main, cur_values, next_values1)
    next_values2 = jnp.where(main, next_values2, cur_values)
    reward1 = jnp.where(main, 0, -r_t + ratio * reward1)
    reward2 = jnp.where(main, -r_t + ratio * reward2, 0)
    xi1 = jnp.where(main, 1, ratio * xi1)
    xi2 = jnp.where(main, ratio * xi2, 1)

    carry = v1, v2, next_values1, next_values2, reward1, reward2, xi1, xi2, \
        last_return1, last_return2, next_q1, next_q2
    return carry, (v, q_t, return_t)


def vtrace_2p0s(
    next_value, ratios, values, rewards, next_dones, mains,
    gamma, rho_min=0.001, rho_max=1.0, c_min=0.001, c_max=1.0, upgo=False,
):
    next_value1 = next_value
    next_value2 = -next_value1
    v1 = return1 = next_q1 = next_value1
    v2 = return2 = next_q2 = next_value2
    reward1 = reward2 = jnp.zeros_like(next_value)
    xi1 = xi2 = jnp.ones_like(next_value)
    carry = v1, v2, next_value1, next_value2, reward1, reward2, xi1, xi2, \
        return1, return2, next_q1, next_q2

    _, (targets, q_estimate, return_t) = jax.lax.scan(
        partial(vtrace_loop, gamma=gamma, rho_min=rho_min, rho_max=rho_max, c_min=c_min, c_max=c_max),
        carry, (ratios, values, next_dones, rewards, mains), reverse=True
    )
    advantages = q_estimate - values
    if upgo:
        advantages += return_t - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages


def truncated_gae_upgo_loop(carry, inp, gamma, gae_lambda):
    lastgaelam1, lastgaelam2, next_value1, next_value2, reward1, reward2, \
        done_used1, done_used2, last_return1, last_return2, next_q1, next_q2 = carry
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

    # UPGO advantage
    last_return1 = jnp.where(real_done1, 0, last_return1)
    last_return2 = jnp.where(real_done2, 0, last_return2)
    last_return1_ = reward1 + gamma * jnp.where(
        next_q1 >= next_value1, last_return1, next_value1)
    last_return2_ = reward2 + gamma * jnp.where(
        next_q2 >= next_value2, last_return2, next_value2)
    next_q1_ = reward1 + gamma * next_value1
    next_q2_ = reward2 + gamma * next_value2
    next_q1 = jnp.where(main1, next_q1_, next_q1)
    next_q2 = jnp.where(main2, next_q2_, next_q1)
    last_return1 = jnp.where(main1, last_return1_, last_return1)
    last_return2 = jnp.where(main2, last_return2_, last_return2)
    returns = jnp.where(main1, last_return1_, last_return2_)

    delta1 = next_q1_ - cur_value
    delta2 = next_q2_ - cur_value
    lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
    lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
    advantages = jnp.where(main1, lastgaelam1_, lastgaelam2_)
    next_value1 = jnp.where(main1, cur_value, next_value1)
    next_value2 = jnp.where(main2, cur_value, next_value2)
    lastgaelam1 = jnp.where(main1, lastgaelam1_, lastgaelam1)
    lastgaelam2 = jnp.where(main2, lastgaelam2_, lastgaelam2)

    carry = lastgaelam1, lastgaelam2, next_value1, next_value2, reward1, reward2, \
        done_used1, done_used2, last_return1, last_return2, next_q1, next_q2
    return carry, (advantages, returns)


def truncated_gae_2p0s(
    next_value, values, rewards, next_dones, mains, gamma, gae_lambda, upgo,
):
    next_value1 = next_value
    next_value2 = -next_value1
    last_return1 = next_q1 = next_value1
    last_return2 = next_q2 = next_value2
    done_used1 = jnp.ones_like(next_dones[-1])
    done_used2 = jnp.ones_like(next_dones[-1])
    reward1 = reward2 = jnp.zeros_like(next_value)
    lastgaelam1 = lastgaelam2 = jnp.zeros_like(next_value)
    carry = lastgaelam1, lastgaelam2, next_value1, next_value2, reward1, reward2, \
        done_used1, done_used2, last_return1, last_return2, next_q1, next_q2

    _, (advantages, returns) = jax.lax.scan(
        partial(truncated_gae_upgo_loop, gamma=gamma, gae_lambda=gae_lambda),
        carry, (values, next_dones, rewards, mains), reverse=True
    )
    if upgo:
        advantages += returns - values
    targets = values + advantages
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