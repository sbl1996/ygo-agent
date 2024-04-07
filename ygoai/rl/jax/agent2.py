from typing import Tuple, Union, Optional, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from ygoai.rl.jax.transformer import EncoderLayer, PositionalEncoding
from ygoai.rl.jax.modules import MLP, make_bin_params, bytes_to_bin, decode_id


default_embed_init = nn.initializers.uniform(scale=0.001)
default_fc_init1 = nn.initializers.uniform(scale=0.001)
default_fc_init2 = nn.initializers.uniform(scale=0.001)


class ActionEncoder(nn.Module):
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
        x_a_act = embed(13, c // div)(x[:, :, 1])
        x_a_yesno = embed(3, c // div)(x[:, :, 2])
        x_a_phase = embed(4, c // div)(x[:, :, 3])
        x_a_cancel = embed(3, c // div)(x[:, :, 4])
        x_a_finish = embed(3, c // div // 2)(x[:, :, 5])
        x_a_position = embed(9, c // div // 2)(x[:, :, 6])
        x_a_option = embed(6, c // div // 2)(x[:, :, 7])
        x_a_number = embed(13, c // div // 2)(x[:, :, 8])
        x_a_place = embed(31, c // div // 2)(x[:, :, 9])
        x_a_attrib = embed(10, c // div // 2)(x[:, :, 10])
        return jnp.concatenate([
            x_a_msg, x_a_act, x_a_yesno, x_a_phase, x_a_cancel, x_a_finish,
            x_a_position, x_a_option, x_a_number, x_a_place, x_a_attrib], axis=-1)


class CardEncoder(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_id, x):
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

        x_id = mlp(
            (c, c // 4), kernel_init=default_fc_init2)(x_id)
        x_id = layer_norm()(x_id)

        x_loc = x1[:, :, 0]
        f_loc = layer_norm()(embed(9, c)(x_loc))

        x_seq = x1[:, :, 1]
        f_seq = layer_norm()(embed(76, c)(x_seq))
        
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

        x_f = jnp.concatenate([
            x_owner, x_position, x_overley, x_attribute,
            x_race, x_level, x_counter, x_negated,
            x_atk, x_def, x_type], axis=-1)
        x_f = layer_norm()(x_f)
        
        f_cards = jnp.concatenate([x_id, x_f], axis=-1)
        f_cards = f_cards + f_loc + f_seq
        return f_cards


class GlobalEncoder(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

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
    num_layers: int = 2
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

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
        action_encoder = ActionEncoder(channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)

        x_cards = x['cards_']
        x_global = x['global_']
        x_actions = x['actions_']
        x_h_actions = x['h_actions_']
        batch_size = x_cards.shape[0]
        
        valid = x_global[:, -1] == 0

        x_id = decode_id(x_cards[:, :, :2].astype(jnp.int32))
        x_id = id_embed(x_id)

        # Cards
        f_cards = CardEncoder(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)(x_id, x_cards[:, :, 2:])
        g_card_embed = self.param(
            'g_card_embed',
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * 0.02,
            (1, c), self.param_dtype)
        f_g_card = jnp.tile(g_card_embed, (batch_size, 1, 1)).astype(f_cards.dtype)
        f_cards = jnp.concatenate([f_g_card, f_cards], axis=1)

        num_heads = max(2, c // 128)
        for _ in range(self.num_layers):
            f_cards = EncoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(f_cards)
        f_cards = layer_norm(dtype=self.dtype)(f_cards)
        f_g_card = f_cards[:, 0]

        # Global
        x_global = GlobalEncoder(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)(x_global)
        x_global = x_global.astype(self.dtype)
        f_global = x_global + MLP((c * 2, c * 2), dtype=self.dtype, param_dtype=self.param_dtype)(x_global)
        f_global = fc_layer(c, dtype=self.dtype)(f_global)
        f_global = layer_norm(dtype=self.dtype)(f_global)

        # History actions
        x_h_actions = x_h_actions.astype(jnp.int32)
        h_mask = x_h_actions[:, :, 2] == 0  # msg == 0
        h_mask = h_mask.at[:, 0].set(False)

        x_h_id = decode_id(x_h_actions[..., :2])
        x_h_id = MLP(
            (c, c), dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=default_fc_init2)(id_embed(x_h_id))

        x_h_a_feats = action_encoder(x_h_actions[:, :, 2:])
        f_h_actions = layer_norm()(x_h_id) + layer_norm()(fc_layer(c, dtype=jnp.float32)(x_h_a_feats))

        f_h_actions = PositionalEncoding()(f_h_actions)
        for _ in range(self.num_layers):
            f_h_actions = EncoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
                f_h_actions, src_key_padding_mask=h_mask)
        f_g_h_actions = layer_norm(dtype=self.dtype)(f_h_actions[:, 0])

        # Actions
        x_actions = x_actions.astype(jnp.int32)

        na_card_embed = self.param(
            'na_card_embed',
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * 0.02,
            (1, c), self.param_dtype)
        f_na_card = jnp.tile(na_card_embed, (batch_size, 1, 1)).astype(f_cards.dtype)
        f_cards = jnp.concatenate([f_na_card, f_cards[:, 1:]], axis=1)

        spec_index = decode_id(x_actions[..., :2])
        B = jnp.arange(batch_size)
        f_a_cards = f_cards[B[:, None], spec_index]
        f_a_cards = fc_layer(c, dtype=self.dtype)(f_a_cards)

        x_a_feats = action_encoder(x_actions[..., 2:])
        x_a_feats = fc_layer(c, dtype=self.dtype)(x_a_feats)
        f_actions = jnp.concatenate([f_a_cards, x_a_feats], axis=-1)
        f_actions = fc_layer(c, dtype=self.dtype)(nn.leaky_relu(f_actions, negative_slope=0.1))
        f_actions = layer_norm(dtype=self.dtype)(f_actions)

        a_mask = x_actions[:, :, 2] == 0
        a_mask = a_mask.at[:, 0].set(False)

        a_mask_ = (1 - a_mask.astype(f_actions.dtype))
        f_g_actions = (f_actions * a_mask_[:, :, None]).sum(axis=1)
        f_g_actions = f_g_actions / a_mask_.sum(axis=1, keepdims=True)

        # State
        f_state = jnp.concatenate([f_g_card, f_global, f_g_h_actions, f_g_actions], axis=-1)
        f_state = MLP((c * 2, c), dtype=self.dtype, param_dtype=self.param_dtype)(f_state)
        f_state = layer_norm(dtype=self.dtype)(f_state)
        
        # TODO: LSTM
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


class Critic(nn.Module):
    channels: Sequence[int] = (128, 128, 128)
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_state):
        f_state = f_state.astype(self.dtype)
        mlp = partial(MLP, dtype=self.dtype, param_dtype=self.param_dtype)
        x = mlp(self.channels, last_lin=False)(f_state)
        x = nn.Dense(1, dtype=jnp.float32, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return x


class PPOAgent(nn.Module):
    channels: int = 128
    num_layers: int = 2
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        c = self.channels
        encoder = Encoder(
            channels=c,
            num_layers=self.num_layers,
            embedding_shape=self.embedding_shape,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        actor = Actor(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)
        critic = Critic(
            channels=[c, c, c], dtype=self.dtype, param_dtype=self.param_dtype)

        f_actions, f_state, mask, valid = encoder(x)
        logits = actor(f_state, f_actions, mask)
        value = critic(f_state)
        return logits, value, valid


class PPOLSTMAgent(nn.Module):
    channels: int = 128
    num_layers: int = 2
    lstm_channels: int = 512
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    multi_step: bool = False

    @nn.compact
    def __call__(self, inputs):
        if self.multi_step:
            # (num_steps * batch_size, ...)
            carry1, carry2, x, done, switch = inputs
            batch_size = carry1[0].shape[0]
            num_steps = done.shape[0] // batch_size
        else:
            carry, x = inputs

        c = self.channels
        encoder = Encoder(
            channels=c,
            num_layers=self.num_layers,
            embedding_shape=self.embedding_shape,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        f_actions, f_state, mask, valid = encoder(x)

        lstm_layer = nn.OptimizedLSTMCell(
            self.lstm_channels, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=nn.initializers.orthogonal(1.0))
        if self.multi_step:
            def body_fn(cell, carry, x, done, switch):
                carry, init_carry = carry
                carry, y = cell(carry, x)
                carry = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), carry)
                carry = jax.tree.map(lambda x, y: jnp.where(switch[:, None], x, y), init_carry, carry)
                return (carry, init_carry), y
            scan = nn.scan(
                body_fn, variable_broadcast='params',
                split_rngs={'params': False})
            f_state, done, switch = jax.tree.map(
                lambda x: jnp.reshape(x, (num_steps, batch_size) + x.shape[1:]), (f_state, done, switch))
            carry, f_state = scan(lstm_layer, (carry1, carry2), f_state, done, switch)
            f_state = f_state.reshape((-1, f_state.shape[-1]))
        else:
            carry, f_state = lstm_layer(carry, f_state)

        actor = Actor(
            channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)
        critic = Critic(
            channels=[c, c, c], dtype=self.dtype, param_dtype=self.param_dtype)

        logits = actor(f_state, f_actions, mask)
        value = critic(f_state)
        return carry, logits, value, valid
