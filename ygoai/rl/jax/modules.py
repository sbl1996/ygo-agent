from typing import Tuple, Union, Optional, Any
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes


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


class GLUMlp(nn.Module):
    intermediate_size: int
    output_size: Optional[int] = None
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs):

        dense = [
            functools.partial(
                nn.DenseGeneral,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            ) for _ in range(3)
        ]
        output_size = self.output_size or inputs.shape[-1]
        g = dense[0](
            features=self.intermediate_size,
            name="gate",
        )(inputs)
        g = nn.silu(g)
        x = g * dense[1](
            features=self.intermediate_size,
            name="up",
        )(inputs)
        x = dense[2](
            features=output_size,
            name="down",
        )(x)
        return x


class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dtype = jnp.promote_types(self.dtype, jnp.float32)
        x = jnp.asarray(x, dtype)
        x = x * jax.lax.rsqrt(jnp.square(x).mean(-1,
                              keepdims=True) + self.epsilon)

        reduced_feature_shape = (x.shape[-1],)
        scale = self.param(
            "scale", nn.initializers.ones, reduced_feature_shape, self.param_dtype
        )
        x = x * scale
        return jnp.asarray(x, self.dtype)


class ReZero(nn.Module):
    channel_wise: bool = False
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        shape = (x.shape[-1],) if self.channel_wise else ()
        scale = self.param("scale", nn.initializers.zeros, shape, self.param_dtype)
        return x * scale


class BatchRenorm(nn.Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation: 
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


    Attributes:
        use_running_average: if True, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
        axis: the feature or non-batch axis of the input.
        momentum: decay rate for the exponential moving average of the batch
            statistics.
        epsilon: a small float added to variance to avoid dividing by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        use_bias:  if True, bias (beta) is added.
        use_scale: if True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
        bias_init: initializer for bias, by default, zero.
        scale_init: initializer for scale, by default, one.
        axis_name: the axis name used to combine batch statistics from multiple
            devices. See `jax.pmap` for a description of axis names (default: None).
        axis_index_groups: groups of axis indices within that named axis
            representing subsets of devices to reduce over (default: None). For
            example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
            examples on the first two and last two devices. See `jax.lax.psum` for
            more details.
        use_fast_variance: If true, use a faster, but less numerically stable,
            calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    scale_init: nn.initializers.Initializer = nn.initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
        Args:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
            Normalized inputs (the same shape as inputs).
        """

        use_running_average = nn.merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats', 'mean', lambda s: jnp.zeros(s, jnp.float32), feature_shape)
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape)

        r_max = self.variable('batch_stats', 'r_max', lambda s: s, 3)
        d_max = self.variable('batch_stats', 'd_max', lambda s: s, 5)
        steps = self.variable('batch_stats', 'steps', lambda s: s, 0)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
