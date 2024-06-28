import functools

import numpy as np
import jax.numpy as jnp
from flax import nnx

from ygoai.rl.jax.nnx.modules import default_kernel_init, default_bias_init, act, GLUMlp, first_from


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


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


def make_apply_rope(head_dim, max_len, dtype):
    cos, sin = precompute_freqs_cis(
        dim=head_dim, end=max_len, dtype=dtype)

    def add_pos(q, k, p=None): return apply_rotary_pos_emb_index(
        q, k, cos, sin, p)
    return add_pos


# from nnx.MultiHeadAttention
class MultiHeadAttention(nnx.Module):
    """Multi-head attention.

    Example usage::

      >>> import flax.linen as nn
      >>> import jax

      >>> layer = nn.MultiHeadAttention(num_heads=8, qkv_features=16)
      >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0), 6)
      >>> shape = (4, 3, 2, 5)
      >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2, shape), jax.random.uniform(key3, shape)
      >>> variables = layer.init(jax.random.key(0), q)

      >>> # different inputs for inputs_q, inputs_k and inputs_v
      >>> out = layer.apply(variables, q, k, v)
      >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k, inputs_v=k)
      >>> out = layer.apply(variables, q, k)
      >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
      >>> out = layer.apply(variables, q)

      >>> attention_kwargs = dict(
      ...     num_heads=8,
      ...     qkv_features=16,
      ...     kernel_init=nn.initializers.ones,
      ...     bias_init=nn.initializers.zeros,
      ...     dropout_rate=0.5,
      ...     deterministic=False,
      ...     )
      >>> class Module(nn.Module):
      ...   attention_kwargs: dict
      ...
      ...   @nn.compact
      ...   def __call__(self, x, dropout_rng=None):
      ...     out1 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
      ...     out2 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
      ...     return out1, out2
      >>> module = Module(attention_kwargs)
      >>> variables = module.init({'params': key1, 'dropout': key2}, q)

      >>> # out1 and out2 are different.
      >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
      >>> # out3 and out4 are different.
      >>> # out1 and out3 are different. out2 and out4 are different.
      >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
      >>> # out1 and out2 are the same.
      >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
      >>> # out1 and out2 are the same as out3 and out4.
      >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg in `.apply`
      >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6}, dropout_rng=key5)

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      out_kernel_init: optional initializer for the kernel of the output Dense layer,
        if None, the kernel_init is used.
      bias_init: initializer for the bias of the Dense layers.
      out_bias_init: optional initializer for the bias of the output Dense layer,
        if None, the bias_init is used.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features=None,
        out_features=None,
        *,
        dtype=None,
        param_dtype=jnp.float32,
        rope=False,
        max_len=2048,
        broadcast_dropout=True,
        dropout_rate=0.0,
        deterministic=None,
        precision=None,
        kernel_init=default_kernel_init,
        out_kernel_init=None,
        bias_init=default_bias_init,
        out_bias_init=None,
        use_bias: bool = True,
        attention_fn=nnx.dot_product_attention,
        # Deprecated, will be removed.
        qkv_dot_general=None,
        out_dot_general=None,
        qkv_dot_general_cls=None,
        out_dot_general_cls=None,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = (
            qkv_features if qkv_features is not None else in_features
        )
        self.out_features = (
            out_features if out_features is not None else in_features
        )
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rope = rope
        self.max_len = max_len
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init
        self.out_kernel_init = out_kernel_init
        self.bias_init = bias_init
        self.out_bias_init = out_bias_init
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.qkv_dot_general = qkv_dot_general
        self.out_dot_general = out_dot_general
        self.qkv_dot_general_cls = qkv_dot_general_cls
        self.out_dot_general_cls = out_dot_general_cls

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f'Memory dimension ({self.qkv_features}) must be divisible by '
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.qkv_features // self.num_heads

        linear_general = functools.partial(
            nnx.LinearGeneral,
            in_features=self.in_features,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        self.query = linear_general(rngs=rngs)
        self.key = linear_general(rngs=rngs)
        self.value = linear_general(rngs=rngs)

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            rngs=rngs,
        )
        self.rngs = rngs if dropout_rate > 0.0 else None

    def __call__(
        self,
        inputs_q,
        inputs_k=None,
        inputs_v=None,
        *,
        mask=None,
        deterministic=None,
        rngs=None,
        sow_weights=False,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape `[batch_sizes..., length, features]`.
          inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
          mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic.
          rngs: container for random number generators to generate the dropout
            mask when `deterministic` is False. The `rngs` container should have a
            `dropout` key.
          sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        if rngs is None:
            rngs = self.rngs

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    '`inputs_k` cannot be None if `inputs_v` is not None. '
                    'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
                    'value to `inputs_k` and leave `inputs_v` as None.'
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f'Incompatible input dimension, got {inputs_q.shape[-1]} '
                f'but module expects {self.in_features}.'
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        if self.rope:
            add_pos = make_apply_rope(
                self.head_dim, self.max_len, self.dtype)
        else:
            def add_pos(q, k, p=None): return (q, k)

        query, key = add_pos(query, key)

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""
                No `deterministic` argument was provided to MultiHeadAttention
                as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided if 'dropout_rng' is not given."
                    )
                dropout_rng = rngs.dropout()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out


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
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class PositionalEncoding(nnx.Module):
    """Adds (optionally learned) positional embeddings to the inputs.
    """

    def __init__(
            self, in_channels, *, max_len=512, learned=False,
            initializer=sinusoidal_init, rngs: nnx.Rngs):
        self.pos_emb_shape = (1, max_len, in_channels)
        self.max_len = max_len
        self.learned = learned

        init = initializer(max_len=max_len)(None, self.pos_emb_shape)
        if learned:
            self.pos_embedding = nnx.Param(init)
        else:
            self.pos_embedding = None

    def __call__(self, x):
        assert x.ndim == 3, (
            'Number of dimensions should be 3, but it is: %d' % x.ndim
        )
        length = x.shape[1]
        if self.pos_embedding is None:
            pos_embedding = sinusoidal_init(max_len=self.max_len)(
                None, self.pos_emb_shape
            )
        else:
            pos_embedding = self.pos_embedding.value
        return x + pos_embedding[:, :length, :]


class MlpBlock(nnx.Module):

    def __init__(
            self, in_channels, channels=None, out_channels=None,
            *, activation="gelu", use_bias=False, dtype=jnp.float32,
            param_dtype=jnp.float32, kernel_init=default_kernel_init,
            bias_init=default_bias_init, rngs: nnx.Rngs):

        self.in_channels = in_channels
        self.channels = channels or 4 * in_channels
        self.out_channels = out_channels or in_channels
        self.activation = activation

        linear = functools.partial(
            nnx.Linear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.fc1 = linear(self.in_channels, self.channels)
        self.fc2 = linear(self.channels, self.out_channels)

    def __call__(self, x):
        x = self.fc1(x)
        x = act(x, self.activation)
        return x


class EncoderLayer(nnx.Module):

    def __init__(
        self, d_model, n_heads, dim_feedforward=None,
        *, llama=False, activation="relu", rope=False, rope_max_len=2048,
        attn_pdrop=0.0, resid_pdrop=0.0, layer_norm_epsilon=1e-6,
        dtype=None, param_dtype=jnp.float32, kernel_init=default_kernel_init,
        bias_init=default_bias_init, use_bias=False, rngs: nnx.Rngs
    ):
        if not llama and rope:
            raise ValueError("RoPE can only be used with llama=True")
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dtype = dtype

        if llama:
            norm = nnx.RMSNorm
            mlp = GLUMlp
        else:
            norm = nnx.LayerNorm
            mlp = functools.partial(MlpBlock, activation=activation)

        self.ln1 = norm(
            d_model, epsilon=layer_norm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.attn = MultiHeadAttention(
            n_heads, d_model, rope=rope, max_len=rope_max_len,
            dtype=dtype, param_dtype=param_dtype,
            use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init,
            dropout_rate=attn_pdrop, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=resid_pdrop, rngs=rngs if resid_pdrop > 0.0 else None)

        self.ln2 = norm(
            d_model, epsilon=layer_norm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.mlp = mlp(
            d_model, channels=dim_feedforward, dtype=dtype, param_dtype=param_dtype,
            use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=resid_pdrop, rngs=rngs if resid_pdrop > 0.0 else None)

    def __call__(
            self, x, attn_scale=None, attn_bias=None,
            output_scale=None, output_bias=None, *, src_key_padding_mask=None):
        x = jnp.asarray(x, self.dtype)

        y = self.ln1(x)
        if src_key_padding_mask is None:
            mask = None
        else:
            mask = ~src_key_padding_mask[:, None, None, :]
        y = self.attn(y, y, y, mask=mask)
        y = self.dropout1(y)
        if attn_scale is not None:
            y = y * attn_scale
        if attn_bias is not None:
            y = y + attn_bias
        x = x + y

        y = self.ln2(x)
        y = self.mlp(y)
        y = self.dropout2(y)
        if output_scale is not None:
            y = y * output_scale
        if output_bias is not None:
            y = y + output_bias
        x = x + y
        return x


class DecoderLayer(nnx.Module):

    def __init__(
        self, d_model, n_heads, dim_feedforward=None,
        *, llama=False, activation="relu", attn_pdrop=0.0, resid_pdrop=0.0,
        layer_norm_epsilon=1e-6, dtype=None, param_dtype=jnp.float32,
        kernel_init=default_kernel_init, bias_init=default_bias_init,
        use_bias=False, rngs: nnx.Rngs
    ):

        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dtype = dtype

        if llama:
            norm = nnx.RMSNorm
            mlp = GLUMlp
        else:
            norm = nnx.LayerNorm
            mlp = functools.partial(MlpBlock, activation=activation)

        self.ln1 = norm(
            d_model, epsilon=layer_norm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(
            n_heads, d_model, dtype=dtype, param_dtype=param_dtype,
            use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init,
            dropout_rate=attn_pdrop, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=resid_pdrop, rngs=rngs if resid_pdrop > 0.0 else None)

        self.ln2 = norm(
            d_model, epsilon=layer_norm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            n_heads, d_model, dtype=dtype, param_dtype=param_dtype,
            use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init,
            dropout_rate=attn_pdrop, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=resid_pdrop, rngs=rngs if resid_pdrop > 0.0 else None)

        self.ln3 = norm(
            d_model, epsilon=layer_norm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.mlp = mlp(
            d_model, channels=dim_feedforward, dtype=dtype, param_dtype=param_dtype,
            use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=resid_pdrop, rngs=rngs if resid_pdrop > 0.0 else None)

    def __call__(
            self, tgt, memory, *, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        y = self.ln1(tgt)
        if tgt_key_padding_mask is None:
            mask = None
        else:
            mask = ~tgt_key_padding_mask[:, None, None, :]
        y = self.self_attn(y, y, y, mask=mask)
        y = self.dropout1(y)
        x = y + tgt

        y = self.ln2(x)
        if memory_key_padding_mask is None:
            mask = None
        else:
            mask = ~memory_key_padding_mask[:, None, None, :]
        y = self.cross_attn(y, memory, memory, mask=mask)
        y = self.dropout2(y)
        x = y + x

        y = self.ln3(x)
        y = self.mlp(y)
        y = self.dropout3(y)
        x = x + y
        return x
