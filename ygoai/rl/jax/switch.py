import jax
import jax.numpy as jnp


def truncated_gae_sep(
    next_value, values, rewards, next_dones, switch, gamma, gae_lambda, upgo
):
    def body_fn(carry, inp):
        boot_value, boot_done, next_value, lastgaelam, next_q, last_return = carry
        next_done, cur_value, reward, switch = inp

        next_done = jnp.where(switch, boot_done, next_done)
        next_value = jnp.where(switch, -boot_value, next_value)
        lastgaelam = jnp.where(switch, 0, lastgaelam)
        next_q = jnp.where(switch, -boot_value * gamma, next_q)
        last_return = jnp.where(switch, -boot_value, last_return)

        discount = gamma * (1.0 - next_done)
        last_return = reward + discount * jnp.where(
            next_q >= next_value, last_return, next_value)
        next_q = reward + discount * next_value
        delta = next_q - cur_value
        lastgaelam = delta + gae_lambda * discount * lastgaelam
        carry = boot_value, boot_done, cur_value, lastgaelam, next_q, last_return
        return carry, (lastgaelam, last_return)

    next_done = next_dones[-1]
    lastgaelam = jnp.zeros_like(next_value)
    next_q = last_return = next_value
    carry = next_value, next_done, next_value, lastgaelam, next_q, last_return

    _, (advantages, returns) = jax.lax.scan(
        body_fn, carry, (next_dones, values, rewards, switch), reverse=True
    )
    targets = values + advantages
    if upgo:
        advantages += returns - values
    targets = jax.lax.stop_gradient(targets)
    return targets, advantages
