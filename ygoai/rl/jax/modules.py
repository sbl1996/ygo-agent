from typing import Tuple, Union, Optional

import jax.numpy as jnp
import flax.linen as nn


def decode_id(x):
    x = x[..., 0] * 256 + x[..., 1]
    return x


def bytes_to_bin(x, points, intervals):
    points = points.astype(x.dtype)
    intervals = intervals.astype(x.dtype)
    x = decode_id(x)
    x = jnp.expand_dims(x, -1)
    return jnp.clip((x - points + intervals) / intervals, 0, 1)


def make_bin_params(x_max=12000, n_bins=32, sig_bins=24):
    x_max1 = 8000
    x_max2 = x_max
    points1 = jnp.linspace(0, x_max1, sig_bins + 1, dtype=jnp.float32)[1:]
    points2 = jnp.linspace(x_max1, x_max2, n_bins - sig_bins + 1, dtype=jnp.float32)[1:]
    points = jnp.concatenate([points1, points2], axis=0)
    intervals = jnp.concatenate([points[0:1], points[1:] - points[:-1]], axis=0)
    return points, intervals



class MLP(nn.Module):
    features: Tuple[int, ...] = (128, 128)
    last_lin: bool = True
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    last_kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x):
        n = len(self.features)
        for i, c in enumerate(self.features):
            if self.last_lin and i == n - 1:
                kernel_init = self.last_kernel_init
            else:
                kernel_init = self.kernel_init
            x = nn.Dense(
                c, dtype=self.dtype, param_dtype=self.param_dtype,
                kernel_init=kernel_init, use_bias=self.use_bias)(x)
            if i < n - 1 or not self.last_lin:
                x = nn.leaky_relu(x, negative_slope=0.1)
        return x