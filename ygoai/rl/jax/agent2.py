from dataclasses import dataclass
from typing import Tuple, Union, Optional, Sequence, Literal
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from ygoai.rl.jax.transformer import EncoderLayer, PositionalEncoding, LlamaEncoderLayer
from ygoai.rl.jax.modules import MLP, GLUMlp, BatchRenorm, make_bin_params, bytes_to_bin, decode_id
from ygoai.rl.jax.rwkv import Rwkv6SelfAttention


default_embed_init = nn.initializers.uniform(scale=0.001)
default_fc_init1 = nn.initializers.uniform(scale=0.001)
default_fc_init2 = nn.initializers.uniform(scale=0.001)


def get_encoder_layer_cls(noam, n_heads, dtype, param_dtype):
    if noam:
        return LlamaEncoderLayer(n_heads, dtype=dtype, param_dtype=param_dtype, rope=False)
    else:
        return EncoderLayer(n_heads, dtype=dtype, param_dtype=param_dtype)


class ActionEncoderV1(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        c = self.channels
        div = 8
        embed = partial(
            nn.Embed, dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=default_embed_init)
        x_a_msg = embed(30, c // div)(x[:, :, 0])
        x_a_act = embed(10, c // div)(x[:, :, 1])
        x_a_finish = embed(3, c // div // 2)(x[:, :, 2])
        x_a_effect = embed(256, c // div * 2)(x[:, :, 3])
        x_a_phase = embed(4, c // div // 2)(x[:, :, 4])
        x_a_position = embed(9, c // div)(x[:, :, 5])
        x_a_number = embed(13, c // div // 2)(x[:, :, 6])
        x_a_place = embed(31, c // div)(x[:, :, 7])
        x_a_attrib = embed(10, c // div // 2)(x[:, :, 8])
        xs = [x_a_msg, x_a_act, x_a_finish, x_a_effect, x_a_phase,
              x_a_position, x_a_number, x_a_place, x_a_attrib]
        return xs


class CardEncoder(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    version: int = 0

    @nn.compact
    def __call__(self, x_id, x, mask):
        assert self.version > 0
        c = self.channels
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype)
        layer_norm = partial(nn.LayerNorm, use_scale=True, use_bias=True)
        embed = partial(
            nn.Embed, dtype=self.dtype, param_dtype=self.param_dtype, embedding_init=default_embed_init)
        fc_embed = partial(nn.Dense, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)

        num_fc = mlp((c // 8,), last_lin=False)
        bin_points, bin_intervals = make_bin_params(n_bins=32)
        num_transform = lambda x: num_fc(bytes_to_bin(x, bin_points, bin_intervals))

        x1 = x[:, :, :10].astype(jnp.int32)
        x2 = x[:, :, 10:].astype(jnp.float32)

        x_loc = x1[:, :, 0]
        x_seq = x1[:, :, 1]

        c_mask = x_loc == 0
        c_mask = c_mask.at[:, 0].set(False)

        x_owner = embed(2, c // 16)(x1[:, :, 2])
        x_position = embed(9, c // 16)(x1[:, :, 3])
        x_overley = embed(2, c // 16)(x1[:, :, 4])
        x_attribute = embed(8, c // 16)(x1[:, :, 5])
        x_race = embed(27, c // 16)(x1[:, :, 6])
        x_level = embed(14, c // 16)(x1[:, :, 7])
        x_counter = embed(16, c // 16)(x1[:, :, 8])
        x_negated = embed(3, c // 16)(x1[:, :, 9])

        x_atk = num_transform(x2[:, :, 0:2])
        x_atk = fc_embed(c // 16, kernel_init=default_fc_init1)(x_atk)
        x_def = num_transform(x2[:, :, 2:4])
        x_def = fc_embed(c // 16, kernel_init=default_fc_init1)(x_def)
        x_type = fc_embed(c // 16 * 2, kernel_init=default_fc_init2)(x2[:, :, 4:])

        x_id = mlp((c,), kernel_init=default_fc_init2)(x_id)
        x_id = jax.nn.swish(x_id)
        f_loc = embed(9, c // 16 * 2)(x_loc)
        f_seq = embed(76, c // 16 * 2)(x_seq)
        feats = [
            x_id, f_loc, f_seq, x_owner, x_position, x_overley, x_attribute,
            x_race, x_level, x_counter, x_negated, x_atk, x_def, x_type]
        if mask is not None:
            assert len(feats) == mask.shape[-1]
            feats = [
                jnp.where(mask[..., i:i+1] == 1, f, f[..., -1:, :])
                for i, f in enumerate(feats)
            ]
        x_cards = jnp.concatenate(feats[1:], axis=-1)
        x_cards = mlp((c,), kernel_init=default_fc_init2)(x_cards)
        x_cards = x_cards * feats[0]
        f_cards = layer_norm()(x_cards)
        return f_cards, c_mask


class GlobalEncoder(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    version: int = 0

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        c = self.channels
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype)
        layer_norm = partial(nn.LayerNorm, use_scale=True, use_bias=True)
        embed = partial(
            nn.Embed, dtype=self.dtype, param_dtype=self.param_dtype, embedding_init=default_embed_init)
        fc_embed = partial(nn.Dense, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)

        count_embed = embed(100, c // 16)
        hand_count_embed = embed(100, c // 16)

        num_fc = mlp((c // 8,), last_lin=False)
        bin_points, bin_intervals = make_bin_params(n_bins=32)
        num_transform = lambda x: num_fc(bytes_to_bin(x, bin_points, bin_intervals))

        x1 = x[:, :4].astype(jnp.float32)
        x2 = x[:, 4:8].astype(jnp.int32)
        x3 = x[:, 8:22].astype(jnp.int32)

        x_lp = fc_embed(c // 4, kernel_init=default_fc_init2)(num_transform(x1[:, 0:2]))
        x_oppo_lp = fc_embed(c // 4, kernel_init=default_fc_init2)(num_transform(x1[:, 2:4]))

        x_turn = embed(20, c // 8)(x2[:, 0])
        x_phase = embed(11, c // 8)(x2[:, 1])
        x_if_first = embed(2, c // 8)(x2[:, 2])
        x_is_my_turn = embed(2, c // 8)(x2[:, 3])
        
        x_cs = count_embed(x3).reshape((batch_size, -1))
        x_my_hand_c = hand_count_embed(x3[:, 1])
        x_op_hand_c = hand_count_embed(x3[:, 8])

        x = jnp.concatenate([
            x_lp, x_oppo_lp, x_turn, x_phase, x_if_first, x_is_my_turn,
            x_cs, x_my_hand_c, x_op_hand_c], axis=-1)
        x = layer_norm()(x)
        return x


class Encoder(nn.Module):
    channels: int = 128
    out_channels: Optional[int] = None
    num_layers: int = 2
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    freeze_id: bool = False
    use_history: bool = True
    card_mask: bool = False
    noam: bool = False
    action_feats: bool = True
    info_mask: bool = False
    version: int = 0

    @nn.compact
    def __call__(self, x):
        c = self.channels
        if self.embedding_shape is None:
            n_embed, embed_dim = 999, 1024
        elif isinstance(self.embedding_shape, int):
            n_embed, embed_dim = self.embedding_shape, 1024
        else:
            n_embed, embed_dim = self.embedding_shape
        n_embed = 1 + n_embed  # 1 (index 0) for unknown

        layer_norm = partial(nn.LayerNorm, use_scale=True, use_bias=True)
        embed = partial(
            nn.Embed, dtype=jnp.float32, param_dtype=self.param_dtype, embedding_init=default_embed_init)
        fc_layer = partial(nn.Dense, use_bias=False, param_dtype=self.param_dtype)

        id_embed = embed(n_embed, embed_dim)
        card_encoder = CardEncoder(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype,
            version=self.version)
        ActionEncoderCls = ActionEncoderV1
        action_encoder = ActionEncoderCls(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)
        
        x_cards = x['cards_']
        x_global = x['global_']
        x_actions = x['actions_']
        x_h_actions = x['h_actions_']
        mask = x['mask_'] if self.info_mask else None
        batch_size = x_global.shape[0]
        
        valid = x_global[:, -1] == 0

        x_id = decode_id(x_cards[:, :, :2].astype(jnp.int32))
        x_id = id_embed(x_id)
        if self.freeze_id:
            x_id = jax.lax.stop_gradient(x_id)
        f_cards, c_mask = card_encoder(x_id, x_cards[:, :, 2:], mask)

        # Cards
        g_card_embed = self.param(
            "g_card_embed",
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * 0.02,
            (1, c), self.param_dtype)
        f_g_card = jnp.tile(g_card_embed, (batch_size, 1, 1)).astype(f_cards.dtype)
        f_cards = jnp.concatenate([f_g_card, f_cards], axis=1)
        if self.card_mask:
            c_mask = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=c_mask.dtype), c_mask], axis=1)
        else:
            c_mask = None

        num_heads = max(2, c // 128)
        for _ in range(self.num_layers):
            f_cards = get_encoder_layer_cls(
                self.noam, num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
                f_cards, src_key_padding_mask=c_mask)
        f_cards = layer_norm(dtype=self.dtype)(f_cards)
        f_g_card = f_cards[:, 0]

        # Global
        x_global = GlobalEncoder(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype, version=self.version)(x_global)
        x_global = x_global.astype(self.dtype)
        if self.version == 2:
            x_global = fc_layer(c, dtype=jnp.float32)(x_global)
            f_global = x_global + GLUMlp(c * 2, dtype=self.dtype, param_dtype=self.param_dtype)(
                layer_norm(dtype=self.dtype)(x_global))
        else:
            f_global = x_global + MLP((c * 2, c * 2), dtype=self.dtype, param_dtype=self.param_dtype)(x_global)
            f_global = fc_layer(c, dtype=self.dtype)(f_global)
        f_global = layer_norm(dtype=self.dtype)(f_global)

        # History actions
        x_h_actions = x_h_actions.astype(jnp.int32)

        h_mask = x_h_actions[:, :, 3] == 0  # msg == 0
        h_mask = h_mask.at[:, 0].set(False)

        x_h_id = decode_id(x_h_actions[..., 1:3])
        x_h_id = id_embed(x_h_id)
        if self.freeze_id:
            x_h_id = jax.lax.stop_gradient(x_h_id)

        x_h_id = fc_layer(c, dtype=jnp.float32)(x_h_id)

        x_h_a_feats = action_encoder(x_h_actions[:, :, 3:12])
        x_h_a_turn = embed(20, c // 2)(x_h_actions[:, :, 12])
        x_h_a_phase = embed(12, c // 2)(x_h_actions[:, :, 13])
        x_h_a_feats.extend([x_h_id, x_h_a_turn, x_h_a_phase])
        x_h_a_feats = jnp.concatenate(x_h_a_feats, axis=-1)
        x_h_a_feats = layer_norm()(x_h_a_feats)
        x_h_a_feats = fc_layer(c, dtype=self.dtype)(x_h_a_feats)

        if self.noam:
            f_h_actions = LlamaEncoderLayer(
                num_heads, dtype=self.dtype, param_dtype=self.param_dtype,
                rope=True, n_positions=64)(x_h_a_feats, src_key_padding_mask=h_mask)
        else:
            x_h_a_feats = PositionalEncoding()(x_h_a_feats)
            f_h_actions = EncoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
                x_h_a_feats, src_key_padding_mask=h_mask)
        f_g_h_actions = layer_norm(dtype=self.dtype)(f_h_actions[:, 0])


        # Actions
        x_actions = x_actions.astype(jnp.int32)

        na_card_embed = self.param(
            'na_card_embed',
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * 0.02,
            (1, c), self.param_dtype)
        f_na_card = jnp.tile(na_card_embed, (batch_size, 1, 1)).astype(f_cards.dtype)
        f_cards = jnp.concatenate([f_na_card, f_cards[:, 1:]], axis=1)

        spec_index = x_actions[..., 0]
        B = jnp.arange(batch_size)
        f_a_cards = f_cards[B[:, None], spec_index]

        x_a_id = decode_id(x_actions[..., 1:3])
        x_a_id = id_embed(x_a_id)
        if self.freeze_id:
            x_a_id = jax.lax.stop_gradient(x_a_id)
        x_a_id = fc_layer(c, dtype=jnp.float32)(x_a_id)

        x_a_feats = action_encoder(x_actions[..., 3:])
        x_a_feats.append(x_a_id)
        x_a_feats = jnp.concatenate(x_a_feats, axis=-1)
        x_a_feats = layer_norm()(x_a_feats)
        x_a_feats = fc_layer(c, dtype=self.dtype)(x_a_feats)
        f_a_cards = fc_layer(c, dtype=self.dtype)(f_a_cards)
        f_actions = jax.nn.silu(f_a_cards) * x_a_feats
        f_actions = fc_layer(c, dtype=self.dtype)(f_actions)
        f_actions = x_a_feats + f_actions

        a_mask = x_actions[:, :, 3] == 0
        a_mask = a_mask.at[:, 0].set(False)

        g_feats = [f_g_card, f_global]
        if self.use_history:
            g_feats.append(f_g_h_actions)

        if self.action_feats:
            f_actions_g = fc_layer(c, dtype=self.dtype)(f_actions)
            a_mask_ = (1 - a_mask.astype(f_actions.dtype))
            f_g_actions = (f_actions_g * a_mask_[:, :, None]).sum(axis=1)
            f_g_actions = f_g_actions / a_mask_.sum(axis=1, keepdims=True)
            g_feats.append(f_g_actions)

        f_state = jnp.concatenate(g_feats, axis=-1)
        oc = self.out_channels or c
        if self.version == 2:
            f_state = GLUMlp(
                intermediate_size=c * 2, output_size=oc,
                dtype=self.dtype, param_dtype=self.param_dtype)(f_state)
        else:
            f_state = MLP((c * 2, oc), dtype=self.dtype, param_dtype=self.param_dtype)(f_state)
        f_state = layer_norm(dtype=self.dtype)(f_state)

        return f_actions, f_state, a_mask, valid


class Actor(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_state, f_actions, mask):
        f_state = f_state.astype(self.dtype)
        f_actions = f_actions.astype(self.dtype)
        c = self.channels
        mlp = partial(MLP, dtype=jnp.float32, param_dtype=self.param_dtype, last_kernel_init=nn.initializers.orthogonal(0.01))
        f_state = mlp((c,), use_bias=True)(f_state)
        logits = jnp.einsum('bc,bnc->bn', f_state, f_actions)
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, big_neg, logits)
        return logits


class FiLMActor(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    noam: bool = False

    @nn.compact
    def __call__(self, f_state, f_actions, mask):
        f_state = f_state.astype(self.dtype)
        f_actions = f_actions.astype(self.dtype)
        c = self.channels
        t = nn.Dense(c * 4, dtype=self.dtype, param_dtype=self.param_dtype)(f_state)
        a_s, a_b, o_s, o_b  = jnp.split(t[:, None, :], 4, axis=-1)

        num_heads = max(2, c // 128)
        f_actions = get_encoder_layer_cls(
            self.noam, num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
            f_actions, mask, a_s, a_b, o_s, o_b)

        logits = nn.Dense(1, dtype=jnp.float32, param_dtype=self.param_dtype,
                          kernel_init=nn.initializers.orthogonal(0.01))(f_actions)[:, :, 0]
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, big_neg, logits)
        return logits


class Critic(nn.Module):
    channels: Sequence[int] = (128, 128, 128)
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_state, train):
        f_state = f_state.astype(self.dtype)
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype)
        x = mlp(self.channels, last_lin=False)(f_state)
        x = nn.Dense(1, dtype=jnp.float32, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return x


class CrossCritic(nn.Module):
    channels: Sequence[int] = (128, 128, 128)
    # dropout_rate: Optional[float] = None
    batch_norm_momentum: float = 0.99
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_state, train):
        x = f_state.astype(self.dtype)
        linear = partial(nn.Dense, dtype=self.dtype, param_dtype=self.param_dtype, use_bias=False)
        BN = partial(
            BatchRenorm, dtype=self.dtype, param_dtype=self.param_dtype,
            momentum=self.batch_norm_momentum, axis_name="local_devices",
            use_running_average=not train)
        x = BN()(x)
        for c in self.channels:
            x = linear(c)(x)
            # if self.use_layer_norm:
            #     x = nn.LayerNorm()(x)
            x = nn.relu()(x)
            # x = nn.leaky_relu(x, negative_slope=0.1)
            x = BN()(x)
        x = nn.Dense(1, dtype=jnp.float32, param_dtype=self.param_dtype)(x)
        return x


class GlobalCritic(nn.Module):
    channels: Sequence[int] = (128, 128)
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, f_state_r1, f_state_r2, f_state, g_cards):
        f_state = jnp.concatenate([f_state_r1, f_state_r2, f_state, g_cards], axis=-1)
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype)
        x = mlp(self.channels, last_lin=True)(f_state)

        # c = self.channels[-1]
        # t = nn.Dense(c * 2, dtype=self.dtype, param_dtype=self.param_dtype)(g_cards)
        # s, b  = jnp.split(t, 2, axis=-1)
        # x = x * s + b
        # x = mlp([c], last_lin=False)(x)

        x = nn.Dense(1, dtype=jnp.float32, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return x


def rnn_step_by_main(rnn_layer, rstate, f_state, done, main, return_state=False):
    if main is not None:
        rstate1, rstate2 = rstate
        rstate = jax.tree.map(lambda x1, x2: jnp.where(main[:, None], x1, x2), rstate1, rstate2)
    rstate, f_state = rnn_layer(rstate, f_state)
    if main is not None:
        rstate1 = jax.tree.map(lambda x, y: jnp.where(main[:, None], x, y), rstate, rstate1)
        rstate2 = jax.tree.map(lambda x, y: jnp.where(main[:, None], y, x), rstate, rstate2)
        rstate = rstate1, rstate2
    if done is not None:
        rstate = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), rstate)
    if return_state:
        return rstate, (f_state, rstate)
    else:
        return rstate, f_state


def rnn_forward_2p(rnn_layer, rstate, f_state, done, switch_or_main, switch=True, return_state=False):
    if switch:
        def body_fn(cell, carry, x, done, switch):
            rstate, init_rstate2 = carry
            rstate, y = cell(rstate, x)
            rstate = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), rstate)
            rstate = jax.tree.map(lambda x, y: jnp.where(switch[:, None], x, y), init_rstate2, rstate)
            return (rstate, init_rstate2), y
    else:
        def body_fn(cell, carry, x, done, main):
            return rnn_step_by_main(cell, carry, x, done, main, return_state)
    scan = nn.scan(
        body_fn, variable_broadcast='params',
        split_rngs={'params': False})
    rstate, f_state = scan(rnn_layer, rstate, f_state, done, switch_or_main)
    return rstate, f_state


def rnn_step(rnn_layer, rstate, f_state, done):
    rstate, f_state = rnn_layer(rstate, f_state)
    if done is not None:
        rstate = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), rstate)
    return rstate, f_state


def rnn_forward(rnn_layer, rstate, f_state, done):
    def body_fn(cell, carry, x, done):
        return rnn_step(cell, carry, x, done)
    scan = nn.scan(
        body_fn, variable_broadcast='params',
        split_rngs={'params': False})
    rstate, f_state = scan(rnn_layer, rstate, f_state, done)
    return rstate, f_state


@dataclass
class EncoderArgs:
    num_layers: int = 1
    """the number of layers for the agent"""
    num_channels: int = 64
    """the number of channels for the agent"""
    use_history: bool = True
    """whether to use history actions as input for agent"""
    card_mask: bool = False
    """whether to mask the padding card as ignored in the transformer"""
    noam: bool = False
    """whether to use Noam architecture for the transformer layer"""
    action_feats: bool = True
    """whether to use action features for the global state"""
    version: int = 0
    """the version of the environment and the agent"""


@dataclass
class ModelArgs(EncoderArgs):
    rnn_channels: int = 256
    """the number of channels for the RNN in the agent"""
    rnn_type: Optional[Literal['lstm', 'gru', 'rwkv', 'none']] = "lstm"
    """the type of RNN to use, None for no RNN"""
    film: bool = False
    """whether to use FiLM for the actor"""
    rnn_shortcut: bool = False
    """whether to use shortcut for the RNN"""
    batch_norm: bool = False
    """whether to use batch normalization for the critic"""
    critic_width: int = 256
    """the width of the critic"""
    critic_depth: int = 3
    """the depth of the critic"""
    rwkv_head_size: int = 32
    """the head size for the RWKV"""


class RNNAgent(nn.Module):
    num_layers: int = 1
    num_channels: int = 64
    rnn_channels: int = 256
    use_history: bool = True
    card_mask: bool = False
    rnn_type: str = 'lstm'
    film: bool = False
    noam: bool = False
    rwkv_head_size: int = 32
    action_feats: bool = True
    rnn_shortcut: bool = False
    batch_norm: bool = False
    critic_width: int = 256
    critic_depth: int = 3
    version: int = 0

    eval: bool = False
    actor: bool = True
    freeze_id: bool = False
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, rstate, done=None, switch_or_main=None, train=False):
        batch_size = jax.tree.leaves(rstate)[0].shape[0]

        c = self.num_channels
        oc = self.rnn_channels if self.rnn_type == 'rwkv' else None
        encoder = Encoder(
            channels=c,
            out_channels=oc,
            num_layers=self.num_layers,
            embedding_shape=self.embedding_shape,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            freeze_id=self.freeze_id,
            use_history=self.use_history,
            card_mask=self.card_mask,
            noam=self.noam,
            action_feats=self.action_feats,
            info_mask=False if self.eval else self.actor,
            version=self.version,
        )

        f_actions, f_state, mask, valid = encoder(x)

        if self.rnn_type in ['lstm']:
            rnn_layer = nn.OptimizedLSTMCell(
                self.rnn_channels, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))
        elif self.rnn_type == 'gru':
            rnn_layer = nn.GRUCell(
                self.rnn_channels, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))
        elif self.rnn_type == 'rwkv':
            num_heads = self.rnn_channels // self.rwkv_head_size
            rnn_layer = Rwkv6SelfAttention(
                num_heads, dtype=self.dtype, param_dtype=self.param_dtype)
        elif self.rnn_type is None:
            rnn_layer = None

        if rnn_layer is None:
            f_state_r = f_state
        else:
            num_steps = f_state.shape[0] // batch_size
            multi_step = num_steps > 1

            if done is None:
                assert not multi_step

            if multi_step:
                f_state_r, done, switch_or_main = jax.tree.map(
                    lambda x: jnp.reshape(x, (num_steps, batch_size) + x.shape[1:]), (f_state, done, switch_or_main))
                if self.actor:
                    rstate, f_state_r = rnn_forward_2p(
                        rnn_layer, rstate, f_state_r, done, switch_or_main, False)
                else:
                    rstate, f_state_r = rnn_forward(
                        rnn_layer, rstate, f_state_r, done)
                f_state_r = f_state_r.reshape((-1, f_state_r.shape[-1]))
            else:
                if self.actor:
                    rstate, f_state_r = rnn_step_by_main(
                        rnn_layer, rstate, f_state, done, switch_or_main)
                else:
                    rstate, f_state_r = rnn_step(
                        rnn_layer, rstate, f_state, done)

        if self.rnn_shortcut:
            f_state_r = jnp.concatenate([f_state, f_state_r], axis=-1)

        if self.actor:
            if self.film:
                actor = FiLMActor(
                    channels=c, dtype=jnp.float32, param_dtype=self.param_dtype, noam=self.noam)
            else:
                actor = Actor(
                    channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)
            logits = actor(f_state_r, f_actions, mask)

            return rstate, logits
        else:
            CriticCls = CrossCritic if self.batch_norm else Critic
            cs = [self.critic_width] * self.critic_depth
            critic = CriticCls(
                channels=cs, dtype=self.dtype, param_dtype=self.param_dtype)
            value = critic(f_state_r, train)
            return rstate, value

    def init_rnn_state(self, batch_size):
        if self.rnn_type in ['lstm']:
            return (
                np.zeros((batch_size, self.rnn_channels)),
                np.zeros((batch_size, self.rnn_channels)),
            )
        elif self.rnn_type == 'gru':
            return np.zeros((batch_size, self.rnn_channels))
        elif self.rnn_type == 'rwkv':
            head_size = self.rwkv_head_size
            num_heads = self.rnn_channels // self.rwkv_head_size
            return (
                np.zeros((batch_size, num_heads*head_size)),
                np.zeros((batch_size, num_heads*head_size*head_size)),
            )
        else:
            return None