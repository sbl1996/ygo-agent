from typing import Tuple, Union, Optional
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from ygoai.rl.jax.modules import MLP, make_bin_params, bytes_to_bin, decode_id
from ygoai.rl.jax.transformer import EncoderLayer, DecoderLayer, PositionalEncoding


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


class Encoder(nn.Module):
    channels: int = 128
    num_card_layers: int = 2
    num_action_layers: int = 2
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

        layer_norm = partial(nn.LayerNorm, use_scale=False, use_bias=False)
        embed = partial(
            nn.Embed, dtype=jnp.float32, param_dtype=self.param_dtype, embedding_init=default_embed_init)
        fc_embed = partial(nn.Dense, use_bias=False, dtype=jnp.float32, param_dtype=self.param_dtype)
        fc_layer = partial(nn.Dense, use_bias=False, dtype=jnp.float32, param_dtype=self.param_dtype)
        
        id_embed = embed(n_embed, embed_dim)
        count_embed = embed(100, c // 16)
        hand_count_embed = embed(100, c // 16)
        
        num_fc = MLP((c // 8,), last_lin=False, dtype=jnp.float32, param_dtype=self.param_dtype)
        bin_points, bin_intervals = make_bin_params(n_bins=32)
        num_transform = lambda x: num_fc(bytes_to_bin(x, bin_points, bin_intervals))
        
        action_encoder = ActionEncoder(channels=c, dtype=jnp.float32, param_dtype=self.param_dtype)
        x_cards = x['cards_']
        x_global = x['global_']
        x_actions = x['actions_']
        batch_size = x_cards.shape[0]
        
        valid = x_global[:, -1] == 0
        
        x_cards_1 = x_cards[:, :, :12].astype(jnp.int32)
        x_cards_2 = x_cards[:, :, 12:].astype(jnp.float32)
        
        x_id = decode_id(x_cards_1[:, :, :2])
        x_id = id_embed(x_id)
        x_id = MLP(
            (c, c // 4), dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=default_fc_init2)(x_id)
        x_id = layer_norm()(x_id)
        
        x_loc = x_cards_1[:, :, 2]
        c_mask = x_loc == 0
        c_mask = c_mask.at[:, 0].set(False)
        f_loc = layer_norm()(embed(9, c)(x_loc))

        x_seq = x_cards_1[:, :, 3]
        f_seq = layer_norm()(embed(76, c)(x_seq))
        
        x_owner = embed(2, c // 16)(x_cards_1[:, :, 4])
        x_position = embed(9, c // 16)(x_cards_1[:, :, 5])
        x_overley = embed(2, c // 16)(x_cards_1[:, :, 6])
        x_attribute = embed(8, c // 16)(x_cards_1[:, :, 7])
        x_race = embed(27, c // 16)(x_cards_1[:, :, 8])
        x_level = embed(14, c // 16)(x_cards_1[:, :, 9])
        x_counter = embed(16, c // 16)(x_cards_1[:, :, 10])
        x_negated = embed(3, c // 16)(x_cards_1[:, :, 11])
        
        x_atk = num_transform(x_cards_2[:, :, 0:2])
        x_atk = fc_embed(c // 16, kernel_init=default_fc_init1)(x_atk)
        x_def = num_transform(x_cards_2[:, :, 2:4])
        x_def = fc_embed(c // 16, kernel_init=default_fc_init1)(x_def)
        x_type = fc_embed(c // 16 * 2, kernel_init=default_fc_init2)(x_cards_2[:, :, 4:])

        x_feat = jnp.concatenate([
            x_owner, x_position, x_overley, x_attribute,
            x_race, x_level, x_counter, x_negated,
            x_atk, x_def, x_type], axis=-1)
        x_feat = layer_norm()(x_feat)
        
        f_cards = jnp.concatenate([x_id, x_feat], axis=-1)
        f_cards = f_cards + f_loc + f_seq
        
        num_heads = max(2, c // 128)
        for _ in range(self.num_card_layers):
            f_cards = EncoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(f_cards)
        na_card_embed = self.param(
            'na_card_embed',
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * 0.02,
            (1, c), self.param_dtype)
        f_na_card = jnp.tile(na_card_embed, (batch_size, 1, 1)).astype(f_cards.dtype)
        f_cards = jnp.concatenate([f_na_card, f_cards], axis=1)
        c_mask = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=c_mask.dtype), c_mask], axis=1)
        f_cards = layer_norm()(f_cards)
        
        x_global_1 = x_global[:, :4].astype(jnp.float32)
        x_g_lp = fc_embed(c // 4, kernel_init=default_fc_init2)(num_transform(x_global_1[:, 0:2]))
        x_g_oppo_lp = fc_embed(c // 4, kernel_init=default_fc_init2)(num_transform(x_global_1[:, 2:4]))
        
        x_global_2 = x_global[:, 4:8].astype(jnp.int32)
        x_g_turn = embed(20, c // 8)(x_global_2[:, 0])
        x_g_phase = embed(11, c // 8)(x_global_2[:, 1])
        x_g_if_first = embed(2, c // 8)(x_global_2[:, 2])
        x_g_is_my_turn = embed(2, c // 8)(x_global_2[:, 3])
        
        x_global_3 = x_global[:, 8:22].astype(jnp.int32)
        x_g_cs = count_embed(x_global_3).reshape((batch_size, -1))
        x_g_my_hand_c = hand_count_embed(x_global_3[:, 1])
        x_g_op_hand_c = hand_count_embed(x_global_3[:, 8])
        
        x_global = jnp.concatenate([
            x_g_lp, x_g_oppo_lp, x_g_turn, x_g_phase, x_g_if_first, x_g_is_my_turn,
            x_g_cs, x_g_my_hand_c, x_g_op_hand_c], axis=-1)
        x_global = layer_norm()(x_global)
        f_global = x_global + MLP((c * 2, c * 2), dtype=jnp.float32, param_dtype=self.param_dtype)(x_global)
        f_global = fc_layer(c)(f_global)
        f_global = layer_norm()(f_global)
        
        f_cards = f_cards + jnp.expand_dims(f_global, 1)
        
        x_actions = x_actions.astype(jnp.int32)
        
        spec_index = decode_id(x_actions[..., :2])
        B = jnp.arange(batch_size)
        f_a_cards = f_cards[B[:, None], spec_index]
        f_a_cards = f_a_cards + fc_layer(c)(layer_norm()(f_a_cards))
        
        x_a_feats = action_encoder(x_actions[..., 2:])
        f_actions = f_a_cards + layer_norm()(x_a_feats)
        
        a_mask = x_actions[:, :, 2] == 0
        a_mask = a_mask.at[:, 0].set(False)
        for _ in range(self.num_action_layers):
            f_actions = DecoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
                f_actions, f_cards,
                tgt_key_padding_mask=a_mask,
                memory_key_padding_mask=c_mask)

        x_h_actions = x['h_actions_'].astype(jnp.int32)
        h_mask = x_h_actions[:, :, 2] == 0  # msg == 0
        h_mask = h_mask.at[:, 0].set(False)

        x_h_id = decode_id(x_h_actions[..., :2])
        x_h_id = MLP(
            (c, c), dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=default_fc_init2)(id_embed(x_h_id))

        x_h_a_feats = action_encoder(x_h_actions[:, :, 2:])
        f_h_actions = layer_norm()(x_h_id) + layer_norm()(x_h_a_feats)

        f_h_actions = PositionalEncoding()(f_h_actions)
        for _ in range(self.num_action_layers):
            f_h_actions = EncoderLayer(num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(
                f_h_actions, src_key_padding_mask=h_mask)

        for _ in range(self.num_action_layers):
            f_actions = DecoderLayer(num_heads, dtype=jnp.float32, param_dtype=self.param_dtype)(
                f_actions, f_h_actions,
                tgt_key_padding_mask=a_mask,
                memory_key_padding_mask=h_mask)
        
        f_actions = layer_norm()(f_actions)
        
        f_s_cards_global = f_cards.mean(axis=1)
        c_mask = 1 - a_mask[:, :, None].astype(f_actions.dtype)
        f_s_actions_ha = (f_actions * c_mask).sum(axis=1) / c_mask.sum(axis=1)
        f_state = jnp.concatenate([f_s_cards_global, f_s_actions_ha], axis=-1)
        return f_actions, f_state, a_mask, valid


class Actor(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_actions, mask):
        c = self.channels
        mlp = partial(MLP, dtype=jnp.float32, param_dtype=self.param_dtype, last_kernel_init=nn.initializers.orthogonal(0.01))
        num_heads = max(2, c // 128)
        f_actions = EncoderLayer(
            num_heads, dtype=jnp.float32, param_dtype=self.param_dtype)(f_actions, src_key_padding_mask=mask)
        logits = mlp((c // 4, 1), use_bias=True)(f_actions)
        logits = logits[..., 0]
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, big_neg, logits)
        return logits


class Critic(nn.Module):
    channels: int = 128
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, f_state):
        c = self.channels
        mlp = partial(MLP, dtype=jnp.float32, param_dtype=self.param_dtype, last_kernel_init=nn.initializers.orthogonal(1.0))
        x = MLP((c // 2, 1), use_bias=True)(f_state)
        return x


class PPOAgent(nn.Module):
    channels: int = 128
    num_card_layers: int = 2
    num_action_layers: int = 2
    embedding_shape: Optional[Union[int, Tuple[int, int]]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        encoder = Encoder(
            channels=self.channels,
            num_card_layers=self.num_card_layers,
            num_action_layers=self.num_action_layers,
            embedding_shape=self.embedding_shape,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        actor = Actor(channels=self.channels, dtype=self.dtype, param_dtype=self.param_dtype)
        critic = Critic(channels=self.channels, dtype=self.dtype, param_dtype=self.param_dtype)
        
        f_actions, f_state, mask, valid = encoder(x)
        logits = actor(f_actions, mask)
        value = critic(f_state)
        return logits, value, valid
