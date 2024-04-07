import jax.numpy as jnp

from ygoai.rl.env import RecordEpisodeStatistics


def masked_mean(x, valid):
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x.sum() / valid.sum()


def masked_normalize(x, valid, epsilon=1e-8):
    x = jnp.where(valid, x, jnp.zeros_like(x))
    n = valid.sum()
    mean = x.sum() / n
    variance = jnp.square(x - mean).sum() / n
    return (x - mean) / jnp.sqrt(variance + epsilon)