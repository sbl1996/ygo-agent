import jax
import jax.numpy as jnp

from flax import struct

from ygoai.rl.env import RecordEpisodeStatistics


def masked_mean(x, valid):
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x.sum() / valid.sum()


def masked_normalize(x, valid, eps=1e-8):
    x = jnp.where(valid, x, jnp.zeros_like(x))
    n = valid.sum()
    mean = x.sum() / n
    variance = jnp.square(x - mean).sum() / n
    return (x - mean) / jnp.sqrt(variance + eps)


def categorical_sample(logits, key):
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
    return action, key


class RunningMeanStd(struct.PyTreeNode):
    """Tracks the mean, variance and count of values."""

    mean: jnp.ndarray = struct.field(pytree_node=True)
    var: jnp.ndarray = struct.field(pytree_node=True)
    count: jnp.ndarray = struct.field(pytree_node=True)

    @classmethod
    def create(cls, shape=()):
        return cls(
            mean=jnp.zeros(shape, "float64"),
            var=jnp.ones(shape, "float64"),
            count=jnp.full(shape, 1e-4, "float64"),
        )

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        mean, var, count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        return self.replace(mean=mean, var=var, count=count)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
