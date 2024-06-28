from typing import Optional, Any
import functools

import jax
import jax.numpy as jnp
from flax import nnx

from flax.nnx.nnx.nn.normalization import _compute_stats, _normalize, _canonicalize_axes
from ygoai.rl.jax.modules import make_bin_params, bytes_to_bin, decode_id

default_kernel_init = nnx.initializers.lecun_normal()
default_bias_init = nnx.initializers.zeros


def first_from(*args, error_msg: str):
    """Return the first non-None argument.

    If all arguments are None, raise a ValueError with the given error message.

    Args:
      *args: the arguments to check
      error_msg: the error message to raise if all arguments are None
    Returns:
      The first non-None argument.
    """
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(error_msg)


def act(x, activation):
    if activation == 'leaky_relu':
        return nnx.leaky_relu(x, negative_slope=0.1)
    elif activation == 'relu':
        return nnx.relu(x)
    elif activation == 'swich' or activation == 'silu':
        return nnx.swish(x)
    elif activation == 'gelu':
        return nnx.gelu(x, approximate=False)
    elif activation == "gelu_new":
        return nnx.gelu(x, approximate=True)
    else:
        raise ValueError(f'Unknown activation: {activation}')


class MLP(nnx.Module):

    def __init__(
            self, in_channels, channels, *, last_lin=True,
            activation='leaky_relu', use_bias=False, dtype=None, param_dtype=jnp.float32,
            kernel_init=default_kernel_init, bias_init=default_bias_init,
            last_kernel_init=default_kernel_init, rngs: nnx.Rngs):
        if isinstance(channels, int):
            channels = [channels]

        self.in_channels = in_channels
        self.channels = channels
        self.last_lin = last_lin
        self.activation = activation
        self.n_layers = len(channels)

        ic = in_channels
        for i, c in enumerate(channels):
            if i == len(channels) - 1 and last_lin:
                l_kernel_init = last_kernel_init
            else:
                l_kernel_init = kernel_init
            layer = nnx.Linear(
                ic, c, dtype=dtype, param_dtype=param_dtype,
                kernel_init=l_kernel_init, bias_init=bias_init,
                use_bias=use_bias, rngs=rngs)
            ic = c
            setattr(self, f'fc{i+1}', layer)

    def __call__(self, x):
        for i in range(self.n_layers):
            x = getattr(self, f'fc{i+1}')(x)
            if i < self.n_layers - 1 or not self.last_lin:
                x = act(x, self.activation)
        return x


class GLUMlp(nnx.Module):

    def __init__(
            self, in_channels, channels, out_channels=None,
            *, use_bias=False, dtype=None, param_dtype=jnp.float32,
            kernel_init=default_kernel_init, bias_init=default_bias_init,
            rngs: nnx.Rngs):

        self.in_channels = in_channels
        self.channels = channels or 2 * in_channels
        self.out_channels = out_channels or in_channels

        linear = functools.partial(
            nnx.Linear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.gate = linear(self.in_channels, self.channels)
        self.up = linear(self.in_channels, self.channels)
        self.down = linear(self.channels, self.out_channels)

    def __call__(self, x):
        g = self.gate(x)
        x = nnx.silu(g) * self.up(x)
        x = self.down(x)
        return x


class BatchRenorm(nnx.Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation: 
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of
        the batch statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over
        the examples on the first two and last two devices. See `jax.lax.psum`
        for more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
        scale_init: nnx.initializers.Initializer = nnx.initializers.ones_init(),
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: nnx.Rngs,
    ):
        feature_shape = (num_features,)
        self.mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
        self.var = nnx.BatchStat(jnp.ones(feature_shape, jnp.float32))
        self.steps = nnx.BatchStat(jnp.zeros((), jnp.int64))

        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = nnx.Param(None)

        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = nnx.Param(None)

        self.num_features = num_features
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

    def __call__(
        self,
        x,
        use_running_average: Optional[bool] = None,
    ):
        """Normalizes the input using batch statistics.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats
            will be used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = first_from(
            use_running_average,
            self.use_running_average,
            error_msg="""
                No `use_running_average` argument was provided to BatchNorm
                as either a __call__ argument, class attribute, or nnx.flag.""",
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        if use_running_average:
            mean, var = self.mean.value, self.var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var

            # The code below is implemented following the Batch Renormalization paper
            ra_mean = self.mean.value
            ra_var = self.var.value
            steps = self.steps.value
            r_max = 3
            d_max = 5
            r = 1
            d = 0
            std = jnp.sqrt(var + self.epsilon)
            ra_std = jnp.sqrt(ra_var + self.epsilon)
            r = jax.lax.stop_gradient(std / ra_std)
            r = jnp.clip(r, 1 / r_max, r_max)
            d = jax.lax.stop_gradient((mean - ra_mean) / ra_std)
            d = jnp.clip(d, -d_max, d_max)
            tmp_var = var / (r**2)
            tmp_mean = mean - d * jnp.sqrt(custom_var) / r

            # Warm up batch renorm for 100_000 steps to build up proper running statistics
            warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
            custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
            custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

            self.mean.value = (
                self.momentum * ra_mean + (1 - self.momentum) * mean
            )
            self.var.value = (
                self.momentum * ra_var + (1 - self.momentum) * var
            )
            self.steps.value = steps + 1

        return _normalize(
            x,
            custom_mean,
            custom_var,
            self.scale.value,
            self.bias.value,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.epsilon,
        )
