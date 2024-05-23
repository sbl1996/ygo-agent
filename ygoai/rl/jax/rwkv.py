from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.typing import (
  Dtype,
)


TIME_MIX_EXTRA_DIM = 32

def hf_rwkv6_linear_attention(receptance, key, value, time_decay, time_first, state):
    # receptance: (seq_length, batch, num_heads*head_size)
    # key: (seq_length, batch, num_heads*head_size)
    # value: (seq_length, batch, num_heads*head_size)
    # time_decay: (seq_length, batch, num_heads*head_size)
    # time_first: (num_heads, head_size)
    # state: (batch, num_heads, head_size, head_size)
    # out: (seq_length, batch, num_heads, head_size)
    if receptance.ndim == 2:
        receptance = receptance[None]
    shape = state.shape
    seq_length, batch, _ = receptance.shape
    num_heads, head_size = time_first.shape
    key = key.reshape(seq_length, batch, num_heads, head_size)
    value = value.reshape(seq_length, batch, num_heads, head_size)
    receptance = receptance.reshape(seq_length, batch, num_heads, head_size)
    state = state.reshape(batch, num_heads, head_size, head_size)
    time_decay = jnp.exp(-jnp.exp(time_decay)).reshape(seq_length, batch, num_heads, head_size)
    time_first = time_first.reshape(num_heads, head_size, 1) # h, n -> h, n, 1

    def body_fn(carry, inp):
        state, tf = carry
        r, k, v, w = inp
        a = k[:, :, :, None] @ v[:, :, None, :]
        out = (r[:, :, None, :] @ (tf * a + state)).squeeze(2)
        state = a + w[:, :, :, None] * state
        return (state, tf), out

    if seq_length == 1:
        (state, _), out = body_fn((state, time_first), (receptance[0], key[0], value[0], time_decay[0]))
        out = out[None, ...]
    else:
        (state, _), out = jax.lax.scan(body_fn, (state, time_first), (receptance, key, value, time_decay))
    out = out.reshape(seq_length, batch, num_heads * head_size)
    state = state.reshape(shape)
    return out, state


def time_p_initilizer(ratio):
    def init_fn(key, shape, dtype):
        w = jnp.arange(shape[-1], dtype=dtype) / shape[-1]
        p = 1.0 - jnp.power(w, ratio)
        p = jnp.broadcast_to(p, shape)
        return p
    return init_fn


def time_decay_init(key, shape, dtype):
    attention_hidden_size = shape[-1]
    ratio_0_to_1 = 0
    decay_speed = [
        -6.0 + 5.0 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        for h in range(attention_hidden_size)
    ]
    w = jnp.array(decay_speed, dtype=dtype)
    return w[None, None, :]


def time_faaaa_init(key, shape, dtype):
    attention_hidden_size = shape[0] * shape[1]
    ratio_0_to_1 = 0
    w = [
        (1.0 - (i / (attention_hidden_size - 1.0))) * ratio_0_to_1 + 0.1 * ((i + 1) % 3 - 1)
        for i in range(attention_hidden_size)
    ]
    w = jnp.array(w, dtype=dtype)
    return w.reshape(shape)


class Rwkv6SelfAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        B, C = inputs.shape

        def time_w(name, shape):
            return self.param(
                name,
                lambda key, shape, dtype: jax.random.uniform(key, shape, dtype, -1e-4, 1e-4),
                shape, self.param_dtype)

        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        shifted, state  = carry

        x = inputs

        xx = shifted - x
        shifted = inputs

        time_maa_x = self.param(
            "time_maa_x", time_p_initilizer(1.0), (1, C), self.param_dtype)
        time_maa_w1 = time_w("time_maa_w1", (C, TIME_MIX_EXTRA_DIM * 5))
        time_maa_w2 = time_w("time_maa_w2", (5, TIME_MIX_EXTRA_DIM, C))
        xxx = x + xx * time_maa_x
        xxx = jnp.tanh(xxx @ time_maa_w1).reshape((B, 5, -1)).transpose((1, 0, 2))
        xxx = (xxx @ time_maa_w2).reshape((5, B, C))

        time_maa_wkvrg = self.param(
            "time_maa_wkvrg", time_p_initilizer(1.0), (5, 1, C), self.param_dtype)
        x = x[None] + xx[None] * (time_maa_wkvrg + xxx)

        time_decay = x[0]
        rkvg = x[1:5]
        w_rkvg = self.param(
            "w_rkvg", nn.initializers.lecun_normal(),
            (4, 1, C, C), self.param_dtype)
        rkvg = rkvg[:, :, None, :] @ w_rkvg
        receptance, key, value, gate = [
            rkvg[i, :, 0] for i in range(4)
        ]

        time_decay_w1 = time_w("time_decay_w1", (C, TIME_MIX_EXTRA_DIM))
        time_decay_w2 = time_w("time_decay_w2", (TIME_MIX_EXTRA_DIM, C))
        time_decay = jnp.tanh(time_decay @ time_decay_w1) @ time_decay_w2

        time_decay_p = self.param(
            "time_decay", time_decay_init, (1, C), self.param_dtype)
        time_decay = time_decay_p + time_decay

        time_faaaa = self.param(
            "time_faaaa", time_faaaa_init, (self.num_heads, C // self.num_heads), self.param_dtype)

        out, state = hf_rwkv6_linear_attention(
            receptance, key, value, time_decay, time_faaaa, state,
        )
        out = out[0]
        out = nn.GroupNorm(
            num_groups=self.num_heads, epsilon=(1e-5)*(8**2))(out)
        out = out * jax.nn.swish(gate)
        out = dense(features=C, name="output")(out)
        return (shifted, state), out


class Rwkv6SelfAttention0(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        shape1 = inputs.shape
        shape2 = carry[0].shape

        if inputs.ndim == 2:
            inputs = inputs[None, ...]
        T, B, C = inputs.shape
        def time_p(name, ratio=1.0):
            return self.param(
                name, time_p_initilizer(ratio), (1, 1, C), self.param_dtype)

        def time_w(name, shape):
            return self.param(
                name,
                lambda key, shape, dtype: jax.random.uniform(key, shape, dtype, -1e-4, 1e-4),
                shape, self.param_dtype)

        dense = partial(
            nn.Dense,
            features=C,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        shifted, state  = carry
        if shifted.ndim == 2:
            shifted = shifted[None, ...]
        if T != 1:
            shifted = jnp.concatenate([
                shifted, inputs[:-1, 0, :]], axis=0)

        x = inputs

        xx = shifted - x
        shifted = inputs[-1].reshape(shape2)

        xxx = x + xx * time_p('time_maa_x')
        time_maa_w1 = time_w("time_maa_w1", (C, TIME_MIX_EXTRA_DIM * 5))
        xxx = jnp.tanh(xxx @ time_maa_w1).reshape((T*B, 5, -1)).transpose((1, 0, 2))
        time_maa_w2 = time_w("time_maa_w2", (5, TIME_MIX_EXTRA_DIM, C))
        xxx = (xxx @ time_maa_w2).reshape((5, T, B, -1))
        mw, mk, mv, mr, mg = [
            x[0] for x in jnp.split(xxx, 5, axis=0)
        ]


        time_decay = x + xx * (time_p("time_maa_w") + mw)
        key = x + xx * (time_p("time_maa_k") + mk)
        value = x + xx * (time_p("time_maa_v") + mv)
        receptance = x + xx * (time_p("time_maa_r", 0.5) + mr)
        gate = x + xx * (time_p("time_maa_g", 0.5) + mg)

        receptance = dense(name="receptance")(receptance)
        key = dense(name="key")(key)
        value = dense(name="value")(value)
        gate = jax.nn.swish(dense(name="gate")(gate))

        time_decay_w1 = time_w("time_decay_w1", (C, TIME_MIX_EXTRA_DIM))
        time_decay_w2 = time_w("time_decay_w2", (TIME_MIX_EXTRA_DIM, C))
        time_decay = jnp.tanh(time_decay @ time_decay_w1) @ time_decay_w2

        time_decay_p = self.param(
            "time_decay", time_decay_init, (1, 1, C), self.param_dtype)
        time_decay = time_decay_p + time_decay

        time_faaaa = self.param(
            "time_faaaa", time_faaaa_init, (self.num_heads, C // self.num_heads), self.param_dtype)

        out, state = hf_rwkv6_linear_attention(
            receptance, key, value, time_decay, time_faaaa, state,
        )
        out = nn.GroupNorm(
            num_groups=self.num_heads, epsilon=(1e-5)*(8**2))(out)
        out = out * gate
        out = dense(name="output")(out)
        out = out.reshape(shape1)
        return (shifted, state), out


class Rwkv6FeedForward(nn.Module):
    intermediate_size: Optional[int] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        assert inputs.ndim == 3, "inputs must have shape (batch, seq_len, features)"
        T, B, C = inputs.shape
        def time_p(name, ratio=1.0):
            return self.param(
                name, time_p_initilizer(ratio), (1, 1, C), self.param_dtype)

        intermediate_size = self.intermediate_size or int((C * 3.5) // 32 * 32)

        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        _shifted, _state, shifted = carry
        if shifted.ndim == 2:
            shifted = shifted[None, ...]
        if T != 1:
            shifted = jnp.concatenate([
                shifted, inputs[:-1, 0, :]], axis=0)

        x = inputs

        xx = shifted - x

        key = x + xx * time_p('time_maa_k')
        receptance = x + xx * time_p('time_maa_r', 0.5)

        key = jnp.square(jax.nn.relu(
            dense(features=intermediate_size,name="key")(key)))
        value = dense(features=C,name="value")(key)
        receptance = jax.nn.sigmoid(
            dense(features=C,name="receptance")(receptance))
        out = value * receptance
        return (_shifted, _state, inputs[-1]), out


class Rwkv6Block(nn.Module):
    num_heads: int
    intermediate_size: Optional[int] = None
    layer_norm_epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        layer_norm = partial(
            nn.LayerNorm, epsilon=self.layer_norm_epsilon)
        x = inputs
        y = layer_norm(name="ln1")(x)
        carry, y = Rwkv6SelfAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(carry, y)
        x = x + y

        y = layer_norm(name="ln2")(x)
        carry, y = Rwkv6FeedForward(
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(carry, y)
        x = x + y
        return carry, x
