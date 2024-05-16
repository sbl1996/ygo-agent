import functools
from typing import Callable, Optional, Sequence, Union, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.linen.dtypes import promote_dtype


Array = Union[jax.Array, Any]
PRNGKey = jax.Array
RNGSequences = Dict[str, PRNGKey]
Dtype = Union[jax.typing.DTypeLike, Any]
Shape = Sequence[int]
PrecisionLike = Union[jax.lax.Precision, str]


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


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


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * \
            np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)
           ] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class PositionalEncoding(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.
    """
    max_len: int = 512
    learned: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
          inputs: input data.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            'Number of dimensions should be 3, but it is: %d' % inputs.ndim
        )
        length = inputs.shape[1]
        pos_emb_shape = (1, self.max_len, inputs.shape[-1])
        initializer = sinusoidal_init(max_len=self.max_len)
        if self.learned:
            pos_embedding = self.param(
                'pos_embedding', initializer, pos_emb_shape
            )
        else:
            pos_embedding = initializer(
                None, pos_emb_shape, None
            )
        pe = pos_embedding[:, :length, :]
        return inputs + pe


def precompute_freqs_cis(
    dim: int, end: int, theta=10000.0, dtype=jnp.float32
):
    # returns:
    #   cos, sin: (end, dim)
    freqs = 1.0 / \
        (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    t = np.arange(end, dtype=np.float32)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    freqs = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(freqs), np.sin(freqs)
    return jnp.array(cos, dtype=dtype), jnp.array(sin, dtype=dtype)


# from chatglm2, different from original rope

def precompute_freqs_cis2(
    dim: int, end: int, theta: float = 10000.0, dtype=jnp.float32
):
    # returns:
    #   cos, sin: (end, dim)
    freqs = 1.0 / \
        (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    t = np.arange(end, dtype=np.float32)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    cos, sin = np.cos(freqs), np.sin(freqs)
    return jnp.array(cos, dtype=dtype), jnp.array(sin, dtype=dtype)


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id=None):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    #   position_id: (batch_size, seq_len)
    # returns:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    if position_id is None:
        q_pos = jnp.arange(q.shape[1])[None, :]
        k_pos = jnp.arange(k.shape[1])[None, :]
    else:
        q_pos = position_id
        k_pos = position_id

    cos_q = jnp.take(cos, q_pos, axis=0)[:, :, None, :]
    sin_q = jnp.take(sin, q_pos, axis=0)[:, :, None, :]
    q = (q * cos_q) + (rotate_half(q) * sin_q)

    cos_k = jnp.take(cos, k_pos, axis=0)[:, :, None, :]
    sin_k = jnp.take(sin, k_pos, axis=0)[:, :, None, :]
    k = (k * cos_k) + (rotate_half(k) * sin_k)
    return q, k


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_index2(q, k, cos, sin, position_id=None):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    #   position_id: (batch_size, seq_len)
    # returns:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    if position_id is None:
        q_pos = jnp.arange(q.shape[1])[None, :]
        k_pos = jnp.arange(k.shape[1])[None, :]
    else:
        q_pos = position_id
        k_pos = position_id

    cos_q = jnp.take(cos, q_pos, axis=0)[:, :, None, :]
    sin_q = jnp.take(sin, q_pos, axis=0)[:, :, None, :]
    q = apply_cos_sin(q, cos_q, sin_q)

    cos_k = jnp.take(cos, k_pos, axis=0)[:, :, None, :]
    sin_k = jnp.take(sin, k_pos, axis=0)[:, :, None, :]
    k = apply_cos_sin(k, cos_k, sin_k)
    return q, k


def apply_cos_sin(x, cos, sin):
    dim = x.shape[-1]
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]
    x1 = x1.reshape(x1.shape[:-1] + (-1, 2))
    x1 = jnp.stack((x1[..., 0] * cos - x1[..., 1] * sin,
                   x1[..., 1] * cos + x1[..., 0] * sin), axis=-1)
    x1 = x1.reshape(x2.shape)
    x = jnp.concatenate((x1, x2), axis=-1)
    return x


def make_apply_rope(head_dim, max_len, dtype, multi_query=False):
    if multi_query:
        cos, sin = precompute_freqs_cis2(
            dim=head_dim // 2, end=max_len, dtype=dtype)

        def add_pos(q, k, p=None): return apply_rotary_pos_emb_index2(
            q, k, cos, sin, p)
    else:
        cos, sin = precompute_freqs_cis(
            dim=head_dim, end=max_len, dtype=dtype)

        def add_pos(q, k, p=None): return apply_rotary_pos_emb_index(
            q, k, cos, sin, p)
    return add_pos


def replicate_for_multi_query(x, num_heads):
    src_num_heads, head_dim = x.shape[-2:]
    x = jnp.repeat(x, num_heads // src_num_heads, axis=-2)
    # x = jnp.expand_dims(x, axis=-2)
    # x = jnp.tile(x, (1, 1, 1, num_heads // src_num_heads, 1))
    # x = jnp.reshape(x, (*x.shape[:2], num_heads, head_dim))
    return x


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
      query: queries for calculating attention with shape of ``[batch..., q_length,
        num_heads, qk_depth_per_head]``.
      key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
      bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``True``.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs and params)
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.

    Returns:
      Output of shape ``[batch..., num_heads, q_length, kv_length]``.
    """
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
        '...qhd,...khd->...hqk', query, key, precision=precision
    )

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias

    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, big_neg, attn_weights)

    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + \
                attn_weights.shape[-2:]
            keep = random.bernoulli(
                dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(
                dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of ``[batch..., q_length,
        num_heads, qk_depth_per_head]``.
      key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
      value: values to be used in attention with shape of ``[batch..., kv_length,
        num_heads, v_depth_per_head]``.
      bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``True``.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs)
      precision: numerical precision of the computation see ``jax.lax.Precision`
        for details.

    Returns:
      Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
    )

    # return weighted sum over values for each query position
    return jnp.einsum(
        '...hqk,...khd->...qhd', attn_weights, value, precision=precision
    )


class MultiheadAttention(nn.Module):
    features: int
    num_heads: int
    max_len: Optional[int] = None
    multi_query_groups: Optional[int] = None
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    broadcast_dropout: bool = False
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_bias_init
    qkv_bias: bool = True
    out_bias: bool = True
    rope: bool = False

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        key_padding_mask: Optional[Array] = None,
        attn_mask: Optional[Array] = None,
    ):
        r"""
        Parameters
        ----------
        query: Array, shape [batch, q_len, features]
            Query features.
        key: Array, shape [batch, kv_len, features]
            Key features.
        value: Array, shape [batch, kv_len, features]
            Value features.
        key_padding_mask: Optional[Array], shape [batch, kv_len]
            Mask to indicate which keys have zero padding.
        attn_mask: Optional[Array], shape [batch, 1, q_len, kv_len]
            Mask to apply to attention scores.

        Returns
        -------
        out: Array, shape [batch, q_len, features]
            Output features.
        """
        features = self.features
        if self.rope:
            assert self.max_len is not None, "max_len must be provided for rope"
        multi_query = self.multi_query_groups is not None
        assert (
            features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = features // self.num_heads

        query = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.qkv_bias,
            axis=-1,
            name="query",
        )(query)

        kv_num_heads = self.num_heads
        if multi_query:
            kv_num_heads = self.multi_query_groups

        kv_dense = [
            functools.partial(
                nn.DenseGeneral,
                features=(kv_num_heads, head_dim),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.qkv_bias,
                axis=-1,
            ) for i in range(2)
        ]

        key = kv_dense[0](name="key")(key)
        value = kv_dense[1](name="value")(value)

        if multi_query:
            key = replicate_for_multi_query(key, self.num_heads)
            value = replicate_for_multi_query(value, self.num_heads)

        if self.rope:
            add_pos = make_apply_rope(
                head_dim, self.max_len, self.dtype, multi_query)
        else:
            def add_pos(q, k, p=None): return (q, k)

        query, key = add_pos(query, key)

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng("dropout")
            deterministic = False
        else:
            deterministic = True

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]

        if attn_mask is not None:
            mask = attn_mask
            if key_padding_mask is not None:
                mask = jnp.logical_or(mask, key_padding_mask)
        else:
            mask = key_padding_mask

        x = dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
        )

        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.out_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out",
        )(x)
        return out


class MlpBlock(nn.Module):
    intermediate_size: Optional[int] = None
    activation: str = "gelu"
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_bias_init

    @nn.compact
    def __call__(self, inputs):
        assert self.activation in [
            "gelu", "gelu_new", "relu"], "activation must be gelu, gelu_new or relu"
        intermediate_size = self.intermediate_size or 4 * inputs.shape[-1]

        dense = [
            functools.partial(
                nn.DenseGeneral,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            ) for _ in range(2)
        ]

        actual_out_dim = inputs.shape[-1]
        x = dense[0](
            features=intermediate_size,
            name="fc_1",
        )(inputs)
        if self.activation == "gelu":
            x = nn.gelu(x, approximate=False)
        elif self.activation == "gelu_new":
            x = nn.gelu(x, approximate=True)
        elif self.activation == "relu":
            x = nn.relu(x)
        x = dense[1](
            features=actual_out_dim,
            name="fc_2",
        )(x)
        return x


class GLUMlpBlock(nn.Module):
    intermediate_size: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_bias_init

    @nn.compact
    def __call__(self, inputs):

        dense = [
            functools.partial(
                nn.DenseGeneral,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            ) for _ in range(3)
        ]

        actual_out_dim = inputs.shape[-1]
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
            features=actual_out_dim,
            name="down",
        )(x)
        return x


class EncoderLayer(nn.Module):
    n_heads: int
    intermediate_size: Optional[int] = None
    activation: str = "relu"
    dtype: Any = None
    param_dtype: Any = jnp.float32
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-6
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init
    deterministic: bool = True

    @nn.compact
    def __call__(
        self, inputs, src_key_padding_mask=None,
        attn_scale=None, attn_bias=None,
        output_scale=None, output_bias=None):
        inputs = jnp.asarray(inputs, self.dtype)
        x = nn.LayerNorm(epsilon=self.layer_norm_epsilon,
                         dtype=self.dtype, name="ln_1")(inputs)
        x = MultiheadAttention(
            features=x.shape[-1],
            num_heads=self.n_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="attn")(x, x, x, key_padding_mask=src_key_padding_mask)
        x = nn.Dropout(rate=self.resid_pdrop)(
            x, deterministic=self.deterministic)

        if attn_scale is not None:
            x = x * attn_scale
        if attn_bias is not None:
            x = x + attn_bias

        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_epsilon,
                         dtype=self.dtype, name="ln_2")(x)
        y = MlpBlock(
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="mlp")(y)
        y = nn.Dropout(rate=self.resid_pdrop)(
            y, deterministic=self.deterministic)

        if output_scale is not None:
            y = y * output_scale
        if output_bias is not None:
            y = y + output_bias
        y = x + y

        return y


class DecoderLayer(nn.Module):
    n_heads: int
    intermediate_size: Optional[int] = None
    activation: str = "relu"
    dtype: Any = None
    param_dtype: Any = jnp.float32
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-6
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init
    deterministic: bool = True

    @nn.compact
    def __call__(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        features = tgt.shape[-1]
        x = nn.LayerNorm(epsilon=self.layer_norm_epsilon,
                         dtype=self.dtype, name="ln_1")(tgt)
        x = MultiheadAttention(
            features=features,
            num_heads=self.n_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="self_attn")(x, x, x, key_padding_mask=tgt_key_padding_mask)
        x = nn.Dropout(rate=self.resid_pdrop)(
            x, deterministic=self.deterministic)
        x = x + tgt

        y = nn.LayerNorm(epsilon=self.layer_norm_epsilon,
                         dtype=self.dtype, name="ln_2")(x)
        y = MultiheadAttention(
            features=features,
            num_heads=self.n_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="cross_attn")(y, memory, memory, key_padding_mask=memory_key_padding_mask)
        y = nn.Dropout(rate=self.resid_pdrop)(
            y, deterministic=self.deterministic)
        y = y + x

        z = nn.LayerNorm(epsilon=self.layer_norm_epsilon,
                         dtype=self.dtype, name="ln_3")(y)
        z = MlpBlock(
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="mlp")(z)
        z = nn.Dropout(rate=self.resid_pdrop)(
            z, deterministic=self.deterministic
        )
        z = y + z
        return z


class LlamaEncoderLayer(nn.Module):
    n_heads: int
    intermediate_size: Optional[int] = None
    n_positions: int = 512
    rope: bool = True
    dtype: Any = None
    param_dtype: Any = jnp.float32
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    rms_norm_eps: float = 1e-6
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init
    deterministic: bool = True

    @nn.compact
    def __call__(
        self, inputs, src_key_padding_mask=None,
        attn_scale=None, attn_bias=None,
        output_scale=None, output_bias=None):
        features = inputs.shape[-1]
        intermediate_size = self.intermediate_size or 2 * features

        x = RMSNorm(epsilon=self.rms_norm_eps,
                    dtype=self.dtype, name="ln_1")(inputs)
        x = MultiheadAttention(
            features=features,
            num_heads=self.n_heads,
            max_len=self.n_positions,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            qkv_bias=False,
            out_bias=False,
            rope=self.rope,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="attn")(x, x, x, key_padding_mask=src_key_padding_mask)
        x = nn.Dropout(rate=self.resid_pdrop)(
            x, deterministic=self.deterministic)

        if attn_scale is not None:
            x = x * attn_scale
        if attn_bias is not None:
            x = x + attn_bias

        x = x + inputs

        y = RMSNorm(epsilon=self.rms_norm_eps,
                    dtype=self.dtype, name="ln_2")(x)
        y = GLUMlpBlock(
            intermediate_size=intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            use_bias=False,
            name="mlp")(y)
        y = nn.Dropout(rate=self.resid_pdrop)(
            y, deterministic=self.deterministic)

        if output_scale is not None:
            y = y * output_scale
        if output_bias is not None:
            y = y + output_bias

        y = x + y
        return y


class LlamaDecoderLayer(nn.Module):
    n_heads: int
    intermediate_size: int
    n_positions: int = 512
    rope: bool = True
    dtype: Any = None
    param_dtype: Any = jnp.float32
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    rms_norm_eps: float = 1e-6
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = default_bias_init
    deterministic: bool = True

    @nn.compact
    def __call__(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        features = tgt.shape[-1]
        x = RMSNorm(epsilon=self.rms_norm_eps,
                    dtype=self.dtype, name="ln_1")(tgt)
        x = MultiheadAttention(
            features=features,
            num_heads=self.n_heads,
            max_len=self.n_positions,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            qkv_bias=False,
            out_bias=False,
            rope=self.rope,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="self_attn")(x, x, x, key_padding_mask=tgt_key_padding_mask)
        x = nn.Dropout(rate=self.resid_pdrop)(
            x, deterministic=self.deterministic)
        x = x + tgt

        y = RMSNorm(epsilon=self.rms_norm_eps,
                    dtype=self.dtype, name="ln_2")(x)
        y = MultiheadAttention(
            features=features,
            num_heads=self.n_heads,
            max_len=self.n_positions,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            qkv_bias=False,
            out_bias=False,
            rope=self.rope,
            dropout_rate=self.attn_pdrop,
            deterministic=self.deterministic,
            name="cross_attn")(y, memory, memory, key_padding_mask=memory_key_padding_mask)
        y = nn.Dropout(rate=self.resid_pdrop)(
            y, deterministic=self.deterministic)
        y = y + x

        z = RMSNorm(epsilon=self.rms_norm_eps,
                    dtype=self.dtype, name="ln_3")(y)
        z = GLUMlpBlock(
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            use_bias=False,
            name="mlp")(z)
        z = nn.Dropout(rate=self.resid_pdrop)(
            z, deterministic=self.deterministic
        )
        z = y + z
        return z
