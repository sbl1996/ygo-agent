from dataclasses import dataclass
from typing import Tuple, Union, Optional, Sequence, Literal
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from ygoai.rl.jax.nnx.transformer import EncoderLayer, PositionalEncoding
from ygoai.rl.jax.nnx.modules import MLP, GLUMlp, BatchRenorm, make_bin_params, bytes_to_bin, decode_id
from ygoai.rl.jax.nnx.rnn import GRUCell, OptimizedLSTMCell


default_embed_init = nnx.initializers.uniform(scale=0.001)
default_fc_init1 = nnx.initializers.uniform(scale=0.001)
default_fc_init2 = nnx.initializers.uniform(scale=0.001)


class ActionEncoder(nnx.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    def __init__(self, channels, *, dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.channels = channels
        self.dtype = dtype
        self.param_dtype = param_dtype

        c = self.channels
        div = 8
        embed = partial(
            nnx.Embed, dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=default_embed_init, rngs=rngs)
        self.e_msg = embed(30, c // div)
        self.e_act = embed(10, c // div)
        self.e_finish = embed(3, c // div // 2)
        self.e_effect = embed(256, c // div * 2)
        self.e_phase = embed(4, c // div // 2)
        self.e_position = embed(9, c // div)
        self.e_number = embed(13, c // div // 2)
        self.e_place = embed(31, c // div)
        self.e_attrib = embed(10, c // div // 2)

    def __call__(self, x):
        x_a_msg = self.e_msg(x[:, :, 0])
        x_a_act = self.e_act(x[:, :, 1])
        x_a_finish = self.e_finish(x[:, :, 2])
        x_a_effect = self.e_effect(x[:, :, 3])
        x_a_phase = self.e_phase(x[:, :, 4])
        x_a_position = self.e_position(x[:, :, 5])
        x_a_number = self.e_number(x[:, :, 6])
        x_a_place = self.e_place(x[:, :, 7])
        x_a_attrib = self.e_attrib(x[:, :, 8])
        return [
            x_a_msg, x_a_act, x_a_finish, x_a_effect, x_a_phase,
            x_a_position, x_a_number, x_a_place, x_a_attrib]


class CardEncoder(nnx.Module):

    def __init__(
        self, channels, id_embed_dim, *, version=1,
        dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.channels = channels
        self.version = version
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.n_bins = 32

        c = self.channels
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)
        norm = partial(
            nnx.LayerNorm, use_scale=True, use_bias=True, dtype=self.dtype, rngs=rngs)
        embed = partial(
            nnx.Embed, dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=default_embed_init, rngs=rngs)
        fc_embed = partial(
            nnx.Linear, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)

        self.fc_num = mlp(self.n_bins, c // 8, last_lin=False)

        self.e_loc = embed(9, c // 16 * 2)
        self.e_seq = embed(76, c // 16 * 2)
        self.e_owner = embed(2, c // 16)
        self.e_position = embed(9, c // 16)
        self.e_overley = embed(2, c // 16)
        self.e_attribute = embed(8, c // 16)
        self.e_race = embed(27, c // 16)
        self.e_level = embed(14, c // 16)
        self.e_counter = embed(16, c // 16)
        self.e_negated = embed(3, c // 16)
        self.fc_atk = fc_embed(c // 8, c // 16, kernel_init=default_fc_init1)
        self.fc_def = fc_embed(c // 8, c // 16, kernel_init=default_fc_init1)
        self.fc_type = fc_embed(25, c // 16 * 2, kernel_init=default_fc_init2)

        self.fc_id = mlp(id_embed_dim, c, kernel_init=default_fc_init2)
        self.fc_cards = mlp(c, c, kernel_init=default_fc_init2)
        self.norm = norm(c)

    def num_transform(self, x):
        bin_points, bin_intervals = make_bin_params(n_bins=32)
        return self.fc_num(bytes_to_bin(x, bin_points, bin_intervals))

    def __call__(self, x_id, x, mask):
        x1 = x[:, :, :10].astype(jnp.int32)
        x2 = x[:, :, 10:].astype(self.dtype)

        c_mask = x1[:, :, 0]
        c_mask = c_mask.at[:, 0].set(False)

        x_loc = self.e_loc(x1[:, :, 0])
        x_seq = self.e_seq(x1[:, :, 1])
        x_owner = self.e_owner(x1[:, :, 2])
        x_position = self.e_position(x1[:, :, 3])
        x_overley = self.e_overley(x1[:, :, 4])
        x_attribute = self.e_attribute(x1[:, :, 5])
        x_race = self.e_race(x1[:, :, 6])
        x_level = self.e_level(x1[:, :, 7])
        x_counter = self.e_counter(x1[:, :, 8])
        x_negated = self.e_negated(x1[:, :, 9])

        x_atk = self.num_transform(x2[:, :, 0:2])
        x_atk = self.fc_atk(x_atk)
        x_def = self.num_transform(x2[:, :, 2:4])
        x_def = self.fc_def(x_def)
        x_type = self.fc_type(x2[:, :, 4:])

        x_id = nnx.swish(self.fc_id(x_id))
        feats_g = [
            x_id, x_loc, x_seq, x_owner, x_position, x_overley, x_attribute,
            x_race, x_level, x_counter, x_negated, x_atk, x_def, x_type]
        if mask is not None:
            assert len(feats_g) == mask.shape[-1]
            feats = [
                jnp.where(mask[..., i:i+1] == 1, f, f[..., -1:, :])
                for i, f in enumerate(feats_g)
            ]
        else:
            feats = feats_g
        x_cards = jnp.concatenate(feats[1:], axis=-1)
        x_cards = self.fc_cards(x_cards)
        x_cards = x_cards * feats[0]
        f_cards = self.norm(x_cards)
        return f_cards, c_mask


class GlobalEncoder(nnx.Module):

    def __init__(
        self, channels, *, version=1, dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.channels = channels
        self.version = version
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.n_bins = 32

        c = self.channels
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)
        norm = partial(
            nnx.LayerNorm, use_scale=True, use_bias=True, dtype=self.dtype, rngs=rngs)
        embed = partial(
            nnx.Embed, dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=default_embed_init, rngs=rngs)
        fc_embed = partial(
            nnx.Linear, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)
        
        self.fc_num = mlp(self.n_bins, c // 8, last_lin=False)

        self.fc_lp = fc_embed(c // 8, c // 4, kernel_init=default_fc_init2)
        self.fc_oppo_lp = fc_embed(c // 8, c // 4, kernel_init=default_fc_init2)
        self.e_turn = embed(20, c // 8)
        self.e_phase = embed(11, c // 8)
        self.e_if_first = embed(2, c // 8)
        self.e_is_my_turn = embed(2, c // 8)
        self.e_count = embed(100, c // 16)
        self.e_hand_count = embed(100, c // 16)

        self.norm = norm(c * 2)
        self.out_channels = c * 2

    def num_transform(self, x):
        bin_points, bin_intervals = make_bin_params(n_bins=32)
        return self.fc_num(bytes_to_bin(x, bin_points, bin_intervals))

    def __call__(self, x):
        x1 = x[:, :4].astype(self.dtype)
        x2 = x[:, 4:8].astype(jnp.int32)
        x3 = x[:, 8:22].astype(jnp.int32)

        x_lp = self.fc_lp(self.num_transform(x1[:, 0:2]))
        x_oppo_lp = self.fc_oppo_lp(self.num_transform(x1[:, 2:4]))

        x_turn = self.e_turn(x2[:, 0])
        x_phase = self.e_phase(x2[:, 1])
        x_if_first = self.e_if_first(x2[:, 2])
        x_is_my_turn = self.e_is_my_turn(x2[:, 3])

        x_cs = self.e_count(x3).reshape((x.shape[0], -1))
        x_my_hand_c = self.e_hand_count(x3[:, 1])
        x_op_hand_c = self.e_hand_count(x3[:, 8])

        x = jnp.concatenate([
            x_lp, x_oppo_lp, x_turn, x_phase, x_if_first, x_is_my_turn,
            x_cs, x_my_hand_c, x_op_hand_c], axis=-1)
        x = self.norm(x)
        return x


def create_id_embed(embedding_shape, dtype, param_dtype, rngs):
    if embedding_shape is None:
        n_embed, embed_dim = 999, 1024
    elif isinstance(embedding_shape, int):
        n_embed, embed_dim = embedding_shape, 1024
    else:
        n_embed, embed_dim = embedding_shape
    n_embed = 1 + n_embed  # 1 (index 0) for unknown
    return nnx.Embed(
        n_embed, embed_dim, dtype=dtype, param_dtype=param_dtype,
        embedding_init=default_embed_init, rngs=rngs)


class Encoder(nnx.Module):

    def __init__(
        self, channels, out_channels=None, num_layers=2, embedding_shape=None,
        *, freeze_id=False, use_history=True, card_mask=False, noam=False,
        action_feats=True, version=1, dtype=None, param_dtype=jnp.float32, 
        rngs: nnx.Rngs):
        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.freeze_id = freeze_id
        self.use_history = use_history
        self.card_mask = card_mask
        self.noam = noam
        self.action_feats = action_feats
        self.version = version

        key = rngs.params()
        c = self.channels

        norm = partial(
            nnx.LayerNorm, use_scale=True, use_bias=True, dtype=dtype, rngs=rngs)
        embed = partial(
            nnx.Embed, dtype=dtype, param_dtype=param_dtype,
            embedding_init=default_embed_init, rngs=rngs)
        fc_layer = partial(
            nnx.Linear, use_bias=False, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        
        self.id_embed = create_id_embed(embedding_shape, dtype, param_dtype, rngs)
        embed_dim = self.id_embed.features
        self.action_encoder = ActionEncoder(
            channels=channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        
        # Cards
        self.card_encoder = CardEncoder(
            channels=channels, id_embed_dim=embed_dim,
            version=version, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        n_heads = max(2, c // 128)
        self.g_card_embed = nnx.Param(
            jax.random.normal(key, (1, 1, c), param_dtype) * 0.02)
        for i in range(num_layers):
            layer = EncoderLayer(
                c, n_heads, llama=self.noam, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            setattr(self, f'card_layer{i+1}', layer)
        self.card_norm = norm(c)
        
        # Global
        self.global_encoder = GlobalEncoder(
            c, version=version, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        in_channels = self.global_encoder.out_channels
        if self.version == 2:
            self.fc_global = fc_layer(in_channels, c, rngs=rngs)
            self.prenorm_global = norm(c)
            self.mlp_global = GLUMlp(
                c, c * 2, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.mlp_global = MLP(
                in_channels, (c * 2, c * 2), dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            self.fc_global = fc_layer(c * 2, c, rngs=rngs)
        self.global_norm = norm(c)

        # History actions
        self.fc_h_id = fc_layer(embed_dim, c, rngs=rngs)
        self.e_h_turn = embed(20, c // 2)
        self.e_h_phase = embed(12, c // 2)
        self.ha_norm_cat = norm(c * 3)
        self.ha_fc = fc_layer(c * 3, c, rngs=rngs)
        if self.noam:
            self.ha_layer = EncoderLayer(
                c, n_heads, llama=True, rope=True, rope_max_len=64,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.ha_pe = PositionalEncoding()
            self.ha_layer = EncoderLayer(
                c, n_heads, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.ha_norm = norm(c)

        # Actions
        self.na_card_embed = nnx.Param(
            jax.random.normal(key, (1, 1, c), param_dtype) * 0.02)
        self.fc_a_id = fc_layer(embed_dim, c, rngs=rngs)
        self.norm_a_cat = norm(c * 2)
        self.fc_a_cat = fc_layer(c * 2, c, rngs=rngs)
        self.fc_a_cards = fc_layer(c, c, rngs=rngs)
        self.fc_a = fc_layer(c, c, rngs=rngs)
        
        # State
        self.fc_a_g = fc_layer(c, c, rngs=rngs)
        oc = self.out_channels or c
        if self.version == 2:
            self.mlp_state = GLUMlp(
                c * 4, c * 2, oc, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.mlp_state = MLP(
                (c * 4, c * 2, oc), dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.state_norm = norm(oc)
    
    def encode_id(self, x):
        x = decode_id(x)
        x = self.id_embed(x)
        if self.freeze_id:
            x = jax.lax.stop_gradient(x)
        return x
    
    def concat_token(self, x, token, mask=None):
        batch_size = x.shape[0]
        token = jnp.tile(token, (batch_size, 1, 1)).astype(x.dtype)
        x = jnp.concatenate([token, x], axis=1)
        if mask is not None:
            mask = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=mask.dtype), mask], axis=1)
        return x, mask
    
    def __call__(self, x):
        x_cards = x['cards_']
        x_global = x['global_']
        x_actions = x['actions_']
        x_h_actions = x['h_actions_']
        mask = x['mask_']
        batch_size = x_global.shape[0]
        
        valid = x_global[:, -1] == 0
        
        # Cards
        x_id = self.encode_id(x_cards[:, :, :2].astype(jnp.int32))
        f_cards, c_mask = self.card_encoder(x_id, x_cards[:, :, 2:], mask)
        f_cards, c_mask = self.concat_token(f_cards, self.g_card_embed.value, c_mask if self.card_mask else None)
        for i in range(self.num_layers):
            f_cards = getattr(self, f'card_layer{i+1}')(
                f_cards, src_key_padding_mask=c_mask)
        f_cards = self.card_norm(f_cards)
        f_g_card = f_cards[:, 0]

        # Global
        x_global = self.global_encoder(x_global)
        if self.version == 2:
            x_global = self.fc_global(x_global)
            f_global = x_global + self.mlp_global(self.prenorm_global(x_global))
        else:
            f_global = x_global + self.mlp_global(x_global)
            f_global = self.fc_global(f_global)
        f_global = self.global_norm(f_global)

        # History actions
        x_h_actions = x_h_actions.astype(jnp.int32)
        h_mask = x_h_actions[:, :, 3] == 0  # msg == 0
        h_mask = h_mask.at[:, 0].set(False)
        
        x_h_id = self.encode_id(x_h_actions[..., 1:3])
        x_h_id = self.fc_h_id(x_h_id)
        
        x_h_a_feats = self.action_encoder(x_h_actions[:, :, 3:12])
        x_h_a_turn = self.e_h_turn(x_h_actions[:, :, 12])
        x_h_a_phase = self.e_h_phase(x_h_actions[:, :, 13])
        x_h_a_feats.extend([x_h_id, x_h_a_turn, x_h_a_phase])
        x_h_a_feats = jnp.concatenate(x_h_a_feats, axis=-1)
        x_h_a_feats = self.ha_norm_cat(x_h_a_feats)
        x_h_a_feats = self.ha_fc(x_h_a_feats)
        if not self.noam:
            x_h_a_feats = self.ha_pe(x_h_a_feats)
        f_h_actions = self.ha_layer(x_h_a_feats, src_key_padding_mask=h_mask)
        f_g_h_actions = self.ha_norm(f_h_actions[:, 0])
        
        # Actions
        x_actions = x_actions.astype(jnp.int32)
        f_cards = self.concat_token(f_cards[:, 1:], self.na_card_embed.value)[0]
        spec_index = x_actions[..., 0]
        f_a_cards = f_cards[jnp.arange(batch_size)[:, None], spec_index]
        
        x_a_id = self.encode_id(x_actions[..., 1:3])
        x_a_id = self.fc_a_id(x_a_id)
        x_a_feats = self.action_encoder(x_actions[..., 3:])
        x_a_feats.append(x_a_id)
        x_a_feats = jnp.concatenate(x_a_feats, axis=-1)
        x_a_feats = self.norm_a_cat(x_a_feats)
        x_a_feats = self.fc_a_cat(x_a_feats)
        f_a_cards = self.fc_a_cards(f_a_cards)
        f_actions = nnx.silu(f_a_cards) * x_a_feats
        f_actions = x_a_feats + self.fc_a(f_actions)
        
        a_mask = x_actions[:, :, 3] == 0
        a_mask = a_mask.at[:, 0].set(False)
        
        # State
        g_feats = [f_g_card, f_global]
        if self.use_history:
            g_feats.append(f_g_h_actions)
        if self.action_feats:
            f_actions_g = self.fc_a_g(f_actions)
            a_mask_ = (1 - a_mask.astype(f_actions.dtype))
            f_g_actions = (f_actions_g * a_mask_[:, :, None]).sum(axis=1)
            f_g_actions = f_g_actions / a_mask_.sum(axis=1, keepdims=True)
            g_feats.append(f_g_actions)
        f_state = jnp.concatenate(g_feats, axis=-1)
        f_state = self.mlp_state(f_state)
        f_state = self.state_norm(f_state)
        return f_actions, f_state, a_mask, valid


class Actor(nnx.Module):
    
    def __init__(
        self, in_channels, channels, *, dtype=None, param_dtype=jnp.float32,
        final_init=nnx.initializers.orthogonal(0.01), rngs: nnx.Rngs):
        self.channels = channels
        self.dtype = dtype
        self.param_dtype = param_dtype

        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype,
                      last_kernel_init=final_init,  rngs=rngs)
        self.mlp = mlp((in_channels, channels), use_bias=True)

    def __call__(self, f_state, f_actions, mask):
        f_state = f_state.astype(self.dtype)
        f_actions = f_actions.astype(self.dtype)
        f_state = self.mlp(f_state)
        logits = jnp.einsum('bc,bnc->bn', f_state, f_actions)
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, big_neg, logits)
        return logits


class FiLMActor(nnx.Module):

    def __init__(
        self, in_channels, channels, *, noam=False, dtype=None, param_dtype=jnp.float32,
        final_init=nnx.initializers.orthogonal(0.01), rngs: nnx.Rngs):
        self.channels = channels
        self.dtype = dtype
        self.param_dtype = param_dtype

        c = self.channels
        self.fc = nnx.Linear(
            in_channels, channels * 4, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)
        n_heads = max(2, channels // 128)
        self.encoder = EncoderLayer(
            channels, n_heads, llama=noam, dtype=self.dtype,
            param_dtype=self.param_dtype, rngs=rngs)
        self.out = nnx.Linear(
            channels, 1, dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=final_init, rngs=rngs)


    def __call__(self, f_state, f_actions, mask):
        f_state = f_state.astype(self.dtype)
        f_actions = f_actions.astype(self.dtype)
        t = self.fc(f_state)
        a_s, a_b, o_s, o_b  = jnp.split(t[:, None, :], 4, axis=-1)

        f_actions = self.encoder(
            f_actions, a_s, a_b, o_s, o_b, src_key_padding_mask=mask)
        logits = self.out(f_actions)[:, :, 0]
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, big_neg, logits)
        return logits


class Critic(nnx.Module):

    def __init__(
        self, in_channels, channels=(128, 128, 128), *,
        dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.channels = channels
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.mlp = MLP(
            in_channels, channels, last_lin=False,
            dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs)

        final_init = nnx.initializers.orthogonal(1.0)
        self.out = nnx.Linear(
            channels[-1], 1, dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=final_init, rngs=rngs)

    def __call__(self, f_state):
        f_state = f_state.astype(self.dtype)
        x = self.mlp(f_state)
        x = self.out(x)
        return x


class CrossCritic(nnx.Module):

    def __init__(
        self, in_channels, channels=(128, 128, 128), bn_momentum=0.99,
        *, dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.channels = channels
        self.dtype = dtype
        self.param_dtype = param_dtype

        linear = partial(
            nnx.Linear, dtype=self.dtype, param_dtype=self.param_dtype,
            use_bias=False, rngs=rngs)
        BN = partial(
            BatchRenorm, dtype=self.dtype, param_dtype=self.param_dtype,
            momentum=bn_momentum, axis_name="local_devices")
        
        ic = in_channels
        self.bn = BN(ic)
        for i, c in enumerate(self.channels):
            setattr(self, f'fc{i + 1}', linear(ic, c))
            setattr(self, f'bn{i + 1}', BN(c))
            ic = c
        self.out = nnx.Linear(
            ic, 1, dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.orthogonal(1.0), rngs=rngs)
    
    def __call__(self, f_state):
        x = f_state.astype(self.dtype)
        x = self.bn(x)
        for i in range(len(self.channels)):
            x = getattr(self, f'fc{i + 1}')(x)
            x = nnx.relu(x)
            x = getattr(self, f'bn{i + 1}')(x)
        x = self.out(x)
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
        def scan_fn(carry, cell, x, done, switch):
            rstate, init_rstate2 = carry
            rstate, y = cell(rstate, x)
            rstate = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), rstate)
            rstate = jax.tree.map(lambda x, y: jnp.where(switch[:, None], x, y), init_rstate2, rstate)
            return (rstate, init_rstate2), y
    else:
        def scan_fn(carry, cell, x, done, main):
            return rnn_step_by_main(cell, carry, x, done, main, return_state)
    rstate, f_state = nnx.scan(
        scan_fn, state_axes={}
    )(rstate, rnn_layer, f_state, done, switch_or_main)
    return rstate, f_state


class Memory(nnx.Module):

    def __init__(
        self, in_channels, channels, rnn_type, switch=False,
        *, dtype=None, param_dtype=jnp.float32, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.channels = channels
        self.rnn_type = rnn_type
        self.switch = switch
        self.dtype = dtype
        self.param_dtype = param_dtype

        if rnn_type == 'lstm':
            self.rnn = OptimizedLSTMCell(
                in_channels, channels, dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.initializers.orthogonal(1.0), rngs=rngs)
        elif rnn_type == 'gru':
            self.rnn = GRUCell(
                in_channels, channels, dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.initializers.orthogonal(1.0), rngs=rngs)
        elif rnn_type == 'rwkv':
            raise NotImplementedError
            # num_heads = channels // 32
            # self.rnn = Rwkv6SelfAttention(
            #     num_heads, dtype=dtype, param_dtype=param_dtype)
        else:
            self.rnn = None

    def __call__(self, rstate, x, done=None, switch_or_main=None):
        if self.rnn is None:
            return rstate, x
        batch_size = jax.tree.leaves(rstate)[0].shape[0]
        num_steps = x.shape[0] // batch_size
        multi_step = num_steps > 1
        if multi_step:
            x, done, switch_or_main = jax.tree.map(
                lambda x: jnp.reshape(x, (num_steps, batch_size) + x.shape[1:]), (x, done, switch_or_main))
            rstate, x = rnn_forward_2p(
                self.rnn, rstate, x, done, switch_or_main, self.switch, return_state=False)
            x = x.reshape((-1, x.shape[-1]))
        else:
            rstate, x = rnn_step_by_main(
                self.rnn, rstate, x, done, switch_or_main, return_state=False)
        return rstate, x

    def init_state(self, batch_size):
        if self.rnn_type == 'lstm':
            return (
                np.zeros((batch_size, self.channels)),
                np.zeros((batch_size, self.channels)),
            )
        elif self.rnn_type == 'gru':
            return np.zeros((batch_size, self.channels))
        elif self.rnn_type == 'rwkv':
            raise NotImplementedError
            # head_size = self.rwkv_head_size
            # num_heads = self.channels // self.rwkv_head_size
            # return (
            #     np.zeros((batch_size, num_heads*head_size)),
            #     np.zeros((batch_size, num_heads*head_size*head_size)),
            # )
        else:
            return None

@dataclass
class EncoderArgs:
    num_layers: int = 2
    """the number of layers for the agent"""
    num_channels: int = 128
    """the number of channels for the agent"""
    use_history: bool = True
    """whether to use history actions as input for agent"""
    card_mask: bool = False
    """whether to mask the padding card as ignored in the transformer"""
    noam: bool = True
    """whether to use Noam architecture for the transformer layer"""
    action_feats: bool = True
    """whether to use action features for the global state"""
    version: int = 2
    """the version of the environment and the agent"""


@dataclass
class ModelArgs(EncoderArgs):
    rnn_channels: int = 512
    """the number of channels for the RNN in the agent"""
    rnn_type: Optional[Literal['lstm', 'gru', 'rwkv', 'none']] = "lstm"
    """the type of RNN to use, None for no RNN"""
    film: bool = True
    """whether to use FiLM for the actor"""
    rnn_shortcut: bool = False
    """whether to use shortcut for the RNN"""
    batch_norm: bool = False
    """whether to use batch normalization for the critic"""
    critic_width: int = 128
    """the width of the critic"""
    critic_depth: int = 3
    """the depth of the critic"""
    rwkv_head_size: int = 32
    """the head size for the RWKV"""


class RNNAgent(nnx.Module):

    def __init__(
        self,
        num_layers: int = 2,
        num_channels: int = 128,
        rnn_channels: int = 512,
        use_history: bool = True,
        card_mask: bool = False,
        rnn_type: str = 'lstm',
        film: bool = True,
        noam: bool = True,
        rwkv_head_size: int = 32,
        action_feats: bool = True,
        rnn_shortcut: bool = False,
        batch_norm: bool = False,
        critic_width: int = 128,
        critic_depth: int = 3,
        version: int = 2,
        q_head: bool = False,
        switch: bool = True,
        freeze_id: bool = False,
        embedding_shape: Optional[Union[int, Tuple[int, int]]] = None,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None
    ):
        self.rnn_shortcut = rnn_shortcut
        self.q_head = q_head

        c = num_channels
        oc = rnn_channels if rnn_type == 'rwkv' else c
        self.encoder = Encoder(
            num_channels,
            out_channels=oc,
            num_layers=num_layers,
            embedding_shape=embedding_shape,
            freeze_id=freeze_id,
            use_history=use_history,
            card_mask=card_mask,
            noam=noam,
            action_feats=action_feats,
            version=version,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.memory = Memory(
            oc, rnn_channels, rnn_type, switch=switch,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        
        ic = rnn_channels + oc if rnn_shortcut else rnn_channels

        actor_init = nnx.initializers.orthogonal(1) if self.q_head else nnx.initializers.orthogonal(0.01)
        actor_cls = partial(FiLMActor, noam=noam) if film else Actor
        self.actor = actor_cls(
            ic, c, dtype=jnp.float32, param_dtype=param_dtype, final_init=actor_init, rngs=rngs)

        critic_cls = CrossCritic if batch_norm else Critic
        cs = [critic_width] * critic_depth
        self.critic = critic_cls(
            ic, channels=cs, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
    
    def __call__(self, x, rstate, done=None, switch_or_main=None):
        f_actions, f_state, mask, valid = self.encoder(x)
        rstate, f_state_r = self.memory(rstate, f_state, done, switch_or_main)
        if self.rnn_shortcut:
            f_state_r = jnp.concatenate([f_state, f_state_r], axis=-1)
        logits = self.actor(f_state_r, f_actions, mask)
        if self.q_head:
            return rstate, logits, valid
        value = self.critic(f_state_r)
        return rstate, logits, value, valid

    def init_rnn_state(self, batch_size):
        return self.memory.init_state(batch_size)