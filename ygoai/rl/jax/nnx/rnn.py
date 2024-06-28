import jax
import jax.numpy as jnp
from flax import nnx


default_kernel_init = nnx.initializers.lecun_normal()
default_bias_init = nnx.initializers.zeros_init()


class OptimizedLSTMCell(nnx.Module):

    def __init__(
        self, in_features, features: int, *,
        gate_fn=nnx.sigmoid, activation_fn=nnx.tanh,
        kernel_init=default_kernel_init, bias_init=default_bias_init,
        recurrent_kernel_init=nnx.initializers.orthogonal(),
        dtype=None, param_dtype=jnp.float32, rngs,
    ):
        self.features = features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn

        self.fc_i = nnx.Linear(
            in_features, 4 * features,
            use_bias=False, kernel_init=kernel_init,
            bias_init=bias_init, dtype=dtype,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.fc_h = nnx.Linear(
            features, 4 * features,
            use_bias=True, kernel_init=recurrent_kernel_init,
            bias_init=bias_init, dtype=dtype,
            param_dtype=param_dtype, rngs=rngs,
        )
    
    def __call__(self, carry, inputs):
        c, h = carry

        dense_i = self.fc_i(inputs)
        dense_h = self.fc_h(h)

        i, f, g, o = jnp.split(dense_i + dense_h, indices_or_sections=4, axis=-1)
        i, f, g, o = self.gate_fn(i), self.gate_fn(f), self.activation_fn(g), self.gate_fn(o)

        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h


class GRUCell(nnx.Module):

    def __init__(
        self, in_features: int, features: int, *,
        gate_fn=nnx.sigmoid, activation_fn=nnx.tanh,
        kernel_init=default_kernel_init, bias_init=default_bias_init,
        recurrent_kernel_init=nnx.initializers.orthogonal(),
        dtype=None, param_dtype=jnp.float32, rngs,
    ):
        self.features = features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn

        self.fc_i = nnx.Linear(
            in_features, 3 * features,
            use_bias=True, kernel_init=kernel_init,
            bias_init=bias_init, dtype=dtype,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.fc_h = nnx.Linear(
            features, 3 * features,
            use_bias=True, kernel_init=recurrent_kernel_init,
            bias_init=bias_init, dtype=dtype,
            param_dtype=param_dtype, rngs=rngs,
        )
    
    def __call__(self, carry, inputs):
        h = carry

        dense_i = self.fc_i(inputs)
        dense_h = self.fc_h(h)

        ir, iz, in_ = jnp.split(dense_i, indices_or_sections=3, axis=-1)
        hr, hz, hn = jnp.split(dense_h, indices_or_sections=3, axis=-1)

        r = self.gate_fn(ir + hr)
        z = self.gate_fn(iz + hz)
        n = self.activation_fn(in_ + r * hn)
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h
