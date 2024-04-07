from functools import partial

import jax
import jax.numpy as jnp
from typing import NamedTuple


class VTraceOutput(NamedTuple):
    q_estimate: jnp.ndarray
    errors: jnp.ndarray


def vtrace(
    v_tm1,
    v_t,
    r_t,
    discount_t,
    rho_tm1,
    lambda_=1.0,
    c_clip_min: float = 0.001,
    c_clip_max: float = 1.007,
    rho_clip_min: float = 0.001,
    rho_clip_max: float = 1.007,
    stop_target_gradients: bool = True,
):
    """

    Args:
      v_tm1: values at time t-1.
      v_t: values at time t.
      r_t: reward at time t.
      discount_t: discount at time t.
      rho_tm1: importance sampling ratios at time t-1.
      lambda_: mixing parameter; a scalar or a vector for timesteps t.
      clip_rho_threshold: clip threshold for importance weights.
      stop_target_gradients: whether or not to apply stop gradient to targets.
    """
    # Clip importance sampling ratios.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    c_tm1 = jnp.clip(rho_tm1, c_clip_min, c_clip_max) * lambda_
    clipped_rhos_tm1 = jnp.clip(rho_tm1, rho_clip_min, rho_clip_max)

    # Compute the temporal difference errors.
    td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

    # Work backwards computing the td-errors.
    def _body(acc, xs):
        td_error, discount, c = xs
        acc = td_error + discount * c * acc
        return acc, acc

    _, errors = jax.lax.scan(
        _body, 0.0, (td_errors, discount_t, c_tm1), reverse=True)

    # Return errors, maybe disabling gradient flow through bootstrap targets.
    errors = jax.lax.select(
        stop_target_gradients,
        jax.lax.stop_gradient(errors + v_tm1) - v_tm1,
        errors)
    targets_tm1 = errors + v_tm1
    q_bootstrap = jnp.concatenate([
        lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
        v_t[-1:],
    ], axis=0)
    q_estimate = r_t + discount_t * q_bootstrap
    return VTraceOutput(q_estimate=q_estimate, errors=errors)


def upgo_return(r_t, v_t, discount_t, stop_target_gradients: bool = True):
    def _body(acc, xs):
        r, v, q, discount = xs
        acc = r + discount * jnp.where(q >= v, acc, v)
        return acc, acc

    # TODO: following alphastar, estimate q_t with one-step target
    # It might be better to use network to estimate q_t
    q_t = r_t[1:] + discount_t[1:] * v_t[1:]  # q[:-1]
    
    _, returns = jax.lax.scan(
        _body, q_t[-1], (r_t[:-1], v_t[:-1], q_t, discount_t[:-1]), reverse=True)

    # Following rlax.vtrace_td_error_and_advantage, part of gradient is reserved
    # Experiments show that where to stop gradient has no impact on the performance
    returns = jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(returns), returns)
    returns = jnp.concatenate([returns, q_t[-1:]], axis=0)
    return returns


def clipped_surrogate_pg_loss(prob_ratios_t, adv_t, mask, epsilon, use_stop_gradient=True):
    adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
    clipped_ratios_t = jnp.clip(prob_ratios_t, 1. - epsilon, 1. + epsilon)
    clipped_objective = jnp.fmin(prob_ratios_t * adv_t, clipped_ratios_t * adv_t)
    return -jnp.mean(clipped_objective * mask)


@partial(jax.jit, static_argnums=(6, 7))
def compute_gae_2p0s(
    next_value, next_done, values, rewards, dones, switch,
    gamma, gae_lambda,
):
    def body_fn(carry, inp):
        boot_value, boot_done, next_value, lastgaelam = carry
        next_done, cur_value, reward, switch = inp

        next_done = jnp.where(switch, boot_done, next_done)
        next_value = jnp.where(switch, -boot_value, next_value)
        lastgaelam = jnp.where(switch, 0, lastgaelam)

        gamma_ = gamma * (1.0 - next_done)
        delta = reward + gamma_ * next_value - cur_value
        lastgaelam = delta + gae_lambda * gamma_ * lastgaelam
        return (boot_value, boot_done, cur_value, lastgaelam), lastgaelam

    dones = jnp.concatenate([dones, next_done[None, :]], axis=0)

    lastgaelam = jnp.zeros_like(next_value)
    carry = next_value, next_done, next_value, lastgaelam

    _, advantages = jax.lax.scan(
        body_fn, carry, (dones[1:], values, rewards, switch), reverse=True
    )
    target_values = advantages + values
    return advantages, target_values


@partial(jax.jit, static_argnums=(6, 7))
def compute_gae_upgo_2p0s(
    next_value, next_done, values, rewards, dones, switch,
    gamma, gae_lambda,
):
    def body_fn(carry, inp):
        boot_value, boot_done, next_value, next_q, last_return, lastgaelam = carry
        next_done, cur_value, reward, switch = inp

        next_done = jnp.where(switch, boot_done, next_done)
        next_value = jnp.where(switch, -boot_value, next_value)
        next_q = jnp.where(switch, -boot_value * gamma, next_q)
        last_return = jnp.where(switch, -boot_value, last_return)
        lastgaelam = jnp.where(switch, 0, lastgaelam)

        gamma_ = gamma * (1.0 - next_done)
        last_return = reward + gamma_ * jnp.where(
            next_q >= next_value, last_return, next_value)
        next_q = reward + gamma_ * next_value
        delta = next_q - cur_value
        lastgaelam = delta + gae_lambda * gamma_ * lastgaelam
        
        carry = boot_value, boot_done, cur_value, next_q, last_return, lastgaelam
        return carry, (lastgaelam, last_return)

    dones = jnp.concatenate([dones, next_done[None, :]], axis=0)

    lastgaelam = jnp.zeros_like(next_value)
    carry = next_value, next_done, next_value, next_value, next_value, lastgaelam

    _, (advantages, returns) = jax.lax.scan(
        body_fn, carry, (dones[1:], values, rewards, switch), reverse=True
    )
    return returns - values, advantages + values


def compute_gae_once(carry, inp, gamma, gae_lambda):
    nextvalues1, nextvalues2, done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2 = carry
    next_done, curvalues, reward, learn = inp
    learn1 = learn
    learn2 = ~learn
    factor = jnp.where(learn1, jnp.ones_like(reward), -jnp.ones_like(reward))
    reward1 = jnp.where(next_done, reward * factor, jnp.where(learn1 & done_used1, 0, reward1))
    reward2 = jnp.where(next_done, reward * -factor, jnp.where(learn2 & done_used2, 0, reward2))
    real_done1 = next_done | ~done_used1
    nextvalues1 = jnp.where(real_done1, 0, nextvalues1)
    lastgaelam1 = jnp.where(real_done1, 0, lastgaelam1)
    real_done2 = next_done | ~done_used2
    nextvalues2 = jnp.where(real_done2, 0, nextvalues2)
    lastgaelam2 = jnp.where(real_done2, 0, lastgaelam2)
    done_used1 = jnp.where(
        next_done, learn1, jnp.where(learn1 & ~done_used1, True, done_used1))
    done_used2 = jnp.where(
        next_done, learn2, jnp.where(learn2 & ~done_used2, True, done_used2))

    delta1 = reward1 + gamma * nextvalues1 - curvalues
    delta2 = reward2 + gamma * nextvalues2 - curvalues
    lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
    lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
    advantages = jnp.where(learn1, lastgaelam1_, lastgaelam2_)
    nextvalues1 = jnp.where(learn1, curvalues, nextvalues1)
    nextvalues2 = jnp.where(learn2, curvalues, nextvalues2)
    lastgaelam1 = jnp.where(learn1, lastgaelam1_, lastgaelam1)
    lastgaelam2 = jnp.where(learn2, lastgaelam2_, lastgaelam2)
    carry = nextvalues1, nextvalues2, done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2
    return carry, advantages


@partial(jax.jit, static_argnums=(7, 8))
def compute_gae(
    next_value, next_done, next_learn,
    values, rewards, dones, learns,
    gamma, gae_lambda,
):
    next_value1 = jnp.where(next_learn, next_value, -next_value)
    next_value2 = -next_value1
    done_used1 = jnp.ones_like(next_done)
    done_used2 = jnp.ones_like(next_done)
    reward1 = jnp.zeros_like(next_value)
    reward2 = jnp.zeros_like(next_value)
    lastgaelam1 = jnp.zeros_like(next_value)
    lastgaelam2 = jnp.zeros_like(next_value)
    carry = next_value1, next_value2, done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2

    dones = jnp.concatenate([dones, next_done[None, :]], axis=0)
    _, advantages = jax.lax.scan(
        partial(compute_gae_once, gamma=gamma, gae_lambda=gae_lambda),
        carry, (dones[1:], values, rewards, learns), reverse=True
    )
    target_values = advantages + values
    return advantages, target_values


def compute_gae_once_upgo(carry, inp, gamma, gae_lambda):
    next_value1, next_value2, next_q1, next_q2, last_return1, last_return2, \
        done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2 = carry
    next_done, curvalues, reward, learn = inp
    learn1 = learn
    learn2 = ~learn
    factor = jnp.where(learn1, jnp.ones_like(reward), -jnp.ones_like(reward))
    reward1 = jnp.where(next_done, reward * factor, jnp.where(learn1 & done_used1, 0, reward1))
    reward2 = jnp.where(next_done, reward * -factor, jnp.where(learn2 & done_used2, 0, reward2))
    real_done1 = next_done | ~done_used1
    next_value1 = jnp.where(real_done1, 0, next_value1)
    last_return1 = jnp.where(real_done1, 0, last_return1)
    lastgaelam1 = jnp.where(real_done1, 0, lastgaelam1)
    real_done2 = next_done | ~done_used2
    next_value2 = jnp.where(real_done2, 0, next_value2)
    last_return2 = jnp.where(real_done2, 0, last_return2)
    lastgaelam2 = jnp.where(real_done2, 0, lastgaelam2)
    done_used1 = jnp.where(
        next_done, learn1, jnp.where(learn1 & ~done_used1, True, done_used1))
    done_used2 = jnp.where(
        next_done, learn2, jnp.where(learn2 & ~done_used2, True, done_used2))

    last_return1_ = reward1 + gamma * jnp.where(
        next_q1 >= next_value1, last_return1, next_value1)
    last_return2_ = reward2 + gamma * jnp.where(
        next_q2 >= next_value2, last_return2, next_value2)
    next_q1_ = reward1 + gamma * next_value1
    next_q2_ = reward2 + gamma * next_value2
    delta1 = next_q1_ - curvalues
    delta2 = next_q2_ - curvalues
    lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
    lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
    returns = jnp.where(learn1, last_return1_, last_return2_)
    advantages = jnp.where(learn1, lastgaelam1_, lastgaelam2_)
    next_value1 = jnp.where(learn1, curvalues, next_value1)
    next_value2 = jnp.where(learn2, curvalues, next_value2)
    lastgaelam1 = jnp.where(learn1, lastgaelam1_, lastgaelam1)
    lastgaelam2 = jnp.where(learn2, lastgaelam2_, lastgaelam2)
    next_q1 = jnp.where(learn1, next_q1_, next_q1)
    next_q2 = jnp.where(learn2, next_q2_, next_q1)
    last_return1 = jnp.where(learn1, last_return1_, last_return1)
    last_return2 = jnp.where(learn2, last_return2_, last_return2)
    carry = next_value1, next_value2, next_q1, next_q2, last_return1, last_return2, \
        done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2
    return carry, (advantages, returns)


@partial(jax.jit, static_argnums=(7, 8))
def compute_gae_upgo(
    next_value, next_done, next_learn,
    values, rewards, dones, learns,
    gamma, gae_lambda,
):
    next_value1 = jnp.where(next_learn, next_value, -next_value)
    next_value2 = -next_value1
    last_return1 = next_q1 = next_value1
    last_return2 = next_q2 = next_value2
    done_used1 = jnp.ones_like(next_done)
    done_used2 = jnp.ones_like(next_done)
    reward1 = jnp.zeros_like(next_value)
    reward2 = jnp.zeros_like(next_value)
    lastgaelam1 = jnp.zeros_like(next_value)
    lastgaelam2 = jnp.zeros_like(next_value)
    carry = next_value1, next_value2, next_q1, next_q2, last_return1, last_return2, \
        done_used1, done_used2, reward1, reward2, lastgaelam1, lastgaelam2

    dones = jnp.concatenate([dones, next_done[None, :]], axis=0)
    _, (advantages, returns) = jax.lax.scan(
        partial(compute_gae_once_upgo, gamma=gamma, gae_lambda=gae_lambda),
        carry, (dones[1:], values, rewards, learns), reverse=True
    )
    return returns - values, advantages + values
