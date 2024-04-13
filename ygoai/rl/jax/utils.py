import jax
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


def categorical_sample(logits, key):
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
    return action, key
