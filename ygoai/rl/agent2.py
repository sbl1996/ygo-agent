import torch
import torch.nn as nn


def bytes_to_bin(x, points, intervals):
    x = x[..., 0] * 256 + x[..., 1]
    x = x.unsqueeze(-1)
    return torch.clamp((x - points + intervals) / intervals, 0, 1)


def make_bin_params(x_max=32000, n_bins=32, sig_bins=24):
    x_max1 = 8000
    x_max2 = x_max
    points1 = torch.linspace(0, x_max1, sig_bins + 1, dtype=torch.float32)[1:]
    points2 = torch.linspace(x_max1, x_max2, n_bins - sig_bins + 1, dtype=torch.float32)[1:]
    points = torch.cat([points1, points2], dim=0)
    intervals = torch.cat([points[0:1], points[1:] - points[:-1]], dim=0)
    return points, intervals


class Encoder(nn.Module):

    def __init__(self, channels=128, num_card_layers=2, num_action_layers=2,
                 num_history_action_layers=2, embedding_shape=None, bias=False, affine=True):
        super(Encoder, self).__init__()
        self.channels = channels
        self.num_history_action_layers = num_history_action_layers

        c = channels
        self.loc_embed = nn.Embedding(9, c)
        self.loc_norm = nn.LayerNorm(c, elementwise_affine=affine)
        self.seq_embed = nn.Embedding(76, c)
        self.seq_norm = nn.LayerNorm(c, elementwise_affine=affine)

        linear = lambda in_features, out_features: nn.Linear(in_features, out_features, bias=bias)

        c_num = c // 8
        n_bins = 32
        self.num_fc = nn.Sequential(
            linear(n_bins, c_num),
            nn.ReLU(),
        )
        bin_points, bin_intervals = make_bin_params(n_bins=n_bins)
        self.bin_points = nn.Parameter(bin_points, requires_grad=False)
        self.bin_intervals = nn.Parameter(bin_intervals, requires_grad=False)

        self.count_embed = nn.Embedding(100, c // 16)
        self.hand_count_embed = nn.Embedding(100, c // 16)

        if embedding_shape is None:
            n_embed, embed_dim = 999, 1024
        elif isinstance(embedding_shape, int):
            n_embed, embed_dim = embedding_shape, 1024
        else:
            n_embed, embed_dim = embedding_shape
        n_embed = 1 + n_embed  # 1 (index 0) for unknown
        self.id_embed = nn.Embedding(n_embed, embed_dim)

        self.id_fc_emb = linear(1024, c // 4)

        self.id_norm = nn.LayerNorm(c // 4, elementwise_affine=False)

        self.owner_embed = nn.Embedding(2, c // 16)
        self.position_embed = nn.Embedding(9, c // 16 * 2)
        self.overley_embed = nn.Embedding(2, c // 16)
        self.attribute_embed = nn.Embedding(8, c // 16)
        self.race_embed = nn.Embedding(27, c // 16)
        self.level_embed = nn.Embedding(14, c // 16)
        self.counter_embed = nn.Embedding(16, c // 16)
        self.type_fc_emb = linear(25, c // 16 * 2)
        self.atk_fc_emb = linear(c_num, c // 16)
        self.def_fc_emb = linear(c_num, c // 16)
        self.feat_norm = nn.LayerNorm(c // 4 * 3, elementwise_affine=affine)

        self.na_card_embed = nn.Parameter(torch.randn(1, c) * 0.02, requires_grad=True)

        num_heads = max(2, c // 128)
        self.card_net = nn.ModuleList([
            nn.TransformerEncoderLayer(
                c, num_heads, c * 4, dropout=0.0, batch_first=True, norm_first=True)
            for i in range(num_card_layers)
        ])

        self.card_norm = nn.LayerNorm(c, elementwise_affine=False)

        self.lp_fc_emb = linear(c_num, c // 4)
        self.oppo_lp_fc_emb = linear(c_num, c // 4)
        self.turn_embed = nn.Embedding(20, c // 8)
        self.phase_embed = nn.Embedding(11, c // 8)
        self.if_first_embed = nn.Embedding(2, c // 8)
        self.is_my_turn_embed = nn.Embedding(2, c // 8)

        self.my_deck_fc_emb = linear(1024, c // 4)

        self.global_norm_pre = nn.LayerNorm(c * 2, elementwise_affine=affine)
        self.global_net = nn.Sequential(
            nn.Linear(c * 2, c * 2),
            nn.ReLU(),
            nn.Linear(c * 2, c * 2),
        )
        self.global_proj = nn.Linear(c * 2, c)
        self.global_norm = nn.LayerNorm(c, elementwise_affine=False)

        divisor = 8
        self.a_msg_embed = nn.Embedding(30, c // divisor)
        self.a_act_embed = nn.Embedding(13, c // divisor)
        self.a_yesno_embed = nn.Embedding(3, c // divisor)
        self.a_phase_embed = nn.Embedding(4, c // divisor)
        self.a_cancel_finish_embed = nn.Embedding(3, c // divisor)
        self.a_position_embed = nn.Embedding(9, c // divisor)
        self.a_option_embed = nn.Embedding(6, c // divisor // 2)
        self.a_number_embed = nn.Embedding(13, c // divisor // 2)
        self.a_place_embed = nn.Embedding(31, c // divisor // 2)
        # TODO: maybe same embedding as attribute_embed
        self.a_attrib_embed = nn.Embedding(10, c // divisor // 2)
        self.a_feat_norm = nn.LayerNorm(c, elementwise_affine=affine)

        self.a_card_norm = nn.LayerNorm(c, elementwise_affine=False)
        self.a_card_proj = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(),
            nn.Linear(c, c),
        )


        self.h_id_fc_emb = linear(1024, c)
        self.h_id_norm = nn.LayerNorm(c, elementwise_affine=False)
        self.h_a_feat_norm = nn.LayerNorm(c, elementwise_affine=False)

        num_heads = max(2, c // 128)
        self.action_card_net = nn.ModuleList([
            nn.TransformerDecoderLayer(
                c, num_heads, c * 4, dropout=0.0, batch_first=True, norm_first=True, bias=False)
            for i in range(num_action_layers)
        ])

        self.action_history_net = nn.ModuleList([
            nn.TransformerDecoderLayer(
                c, num_heads, c * 4, dropout=0.0, batch_first=True, norm_first=True, bias=False)
            for i in range(num_history_action_layers)
        ])

        self.action_norm = nn.LayerNorm(c, elementwise_affine=False)

        self.init_embeddings()

    def init_embeddings(self, scale=0.0001):
        for n, m in self.named_modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -scale, scale)
            elif n in ["atk_fc_emb", "def_fc_emb"]:
                nn.init.uniform_(m.weight, -scale * 10, scale * 10)
            elif n in ["lp_fc_emb", "oppo_lp_fc_emb"]:
                nn.init.uniform_(m.weight, -scale, scale)
            elif "fc_emb" in n:
                nn.init.uniform_(m.weight, -scale, scale)

    def load_embeddings(self, embeddings):
        weight = self.id_embed.weight
        embeddings = torch.from_numpy(embeddings).to(dtype=weight.dtype, device=weight.device)
        unknown_embed = embeddings.mean(dim=0, keepdim=True)
        embeddings = torch.cat([unknown_embed, embeddings], dim=0)
        weight.data.copy_(embeddings)

    def freeze_embeddings(self):
        self.id_embed.weight.requires_grad = False

    def num_transform(self, x):
        return self.num_fc(bytes_to_bin(x, self.bin_points, self.bin_intervals))

    def encode_action_(self, x):
        x_a_msg = self.a_msg_embed(x[:, :, 0])
        x_a_act = self.a_act_embed(x[:, :, 1])
        x_a_yesno = self.a_yesno_embed(x[:, :, 2])
        x_a_phase = self.a_phase_embed(x[:, :, 3])
        x_a_cancel = self.a_cancel_finish_embed(x[:, :, 4])
        x_a_position = self.a_position_embed(x[:, :, 5])
        x_a_option = self.a_option_embed(x[:, :, 6])
        x_a_number = self.a_number_embed(x[:, :, 7])
        x_a_place = self.a_place_embed(x[:, :, 8])
        x_a_attrib = self.a_attrib_embed(x[:, :, 9])
        return x_a_msg, x_a_act, x_a_yesno, x_a_phase, x_a_cancel, x_a_position, x_a_option, x_a_number, x_a_place, x_a_attrib

    def get_action_card_(self, x, f_cards):
        b, n, c = x.shape
        m = c // 2
        spec_index = x.view(b, n, m, 2)
        spec_index = spec_index[..., 0] * 256 + spec_index[..., 1]
        mask = spec_index != 0
        mask[:, :, 0] = True

        spec_index = spec_index.view(b, -1)
        B = torch.arange(b, device=spec_index.device)
        f_a_actions = f_cards[B[:, None], spec_index]
        f_a_actions = f_a_actions.view(b, n, m, -1)

        f_a_actions = (f_a_actions * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True)
        return f_a_actions

    def get_h_action_card_(self, x):
        b, n, _ = x.shape
        x_ids = x.view(b, n, -1, 2)
        x_ids = x_ids[..., 0] * 256 + x_ids[..., 1]

        mask = x_ids != 0
        mask[:, :, 0] = True

        x_ids = self.id_embed(x_ids)
        x_ids = self.h_id_fc_emb(x_ids)

        x_ids = (x_ids * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True)
        return x_ids

    def encode_card_id(self, x):
        x_id = self.id_embed(x)
        x_id = self.id_fc_emb(x_id)
        x_id = self.id_norm(x_id)
        return x_id

    def encode_card_feat1(self, x1):
        x_owner = self.owner_embed(x1[:, :, 2])
        x_position = self.position_embed(x1[:, :, 3])
        x_overley = self.overley_embed(x1[:, :, 4])
        x_attribute = self.attribute_embed(x1[:, :, 5])
        x_race = self.race_embed(x1[:, :, 6])
        x_level = self.level_embed(x1[:, :, 7])
        x_counter = self.counter_embed(x1[:, :, 8])
        return x_owner, x_position, x_overley, x_attribute, x_race, x_level, x_counter
    
    def encode_card_feat2(self, x2):
        x_atk = self.num_transform(x2[:, :, 0:2])
        x_atk = self.atk_fc_emb(x_atk)
        x_def = self.num_transform(x2[:, :, 2:4])
        x_def = self.def_fc_emb(x_def)
        x_type = self.type_fc_emb(x2[:, :, 4:])
        return x_atk, x_def, x_type

    def encode_global(self, x):
        x_global_1 = x[:, :4].float()
        x_g_lp = self.lp_fc_emb(self.num_transform(x_global_1[:, 0:2]))
        x_g_oppo_lp = self.oppo_lp_fc_emb(self.num_transform(x_global_1[:, 2:4]))

        x_global_2 = x[:, 4:8].long()
        x_g_turn = self.turn_embed(x_global_2[:, 0])
        x_g_phase = self.phase_embed(x_global_2[:, 1])
        x_g_if_first = self.if_first_embed(x_global_2[:, 2])
        x_g_is_my_turn = self.is_my_turn_embed(x_global_2[:, 3])

        x_global_3 = x[:, 8:22].long()
        x_g_cs = self.count_embed(x_global_3).flatten(1)
        x_g_my_hand_c = self.hand_count_embed(x_global_3[:, 1])
        x_g_op_hand_c = self.hand_count_embed(x_global_3[:, 8])

        x_global = torch.cat([
            x_g_lp, x_g_oppo_lp, x_g_turn, x_g_phase, x_g_if_first, x_g_is_my_turn,
            x_g_cs, x_g_my_hand_c, x_g_op_hand_c], dim=-1)
        return x_global

    def forward(self, x):
        x_cards = x['cards_']
        x_global = x['global_']
        x_actions = x['actions_']
        
        x_card_ids = x_cards[:, :, :2].long()
        x_card_ids = x_card_ids[..., 0] * 256 + x_card_ids[..., 1]

        x_cards_1 = x_cards[:, :, 2:11].long()
        x_cards_2 = x_cards[:, :, 11:].to(torch.float32)

        x_id = self.encode_card_id(x_card_ids)
        f_loc = self.loc_norm(self.loc_embed(x_cards_1[:, :, 0]))
        f_seq = self.seq_norm(self.seq_embed(x_cards_1[:, :, 1]))

        x_feat1 = self.encode_card_feat1(x_cards_1)
        x_feat2 = self.encode_card_feat2(x_cards_2)

        x_feat = torch.cat([*x_feat1, *x_feat2], dim=-1)
        x_feat = self.feat_norm(x_feat)

        f_cards = torch.cat([x_id, x_feat], dim=-1)
        f_cards = f_cards + f_loc + f_seq

        f_na_card = self.na_card_embed.expand(f_cards.shape[0], -1, -1)
        f_cards = torch.cat([f_na_card, f_cards], dim=1)

        for layer in self.card_net:
            f_cards = layer(f_cards)
        f_cards = self.card_norm(f_cards)
        
        x_global = self.encode_global(x_global)
        x_global = self.global_norm_pre(x_global)
        f_global = x_global + self.global_net(x_global)
        f_global = self.global_proj(f_global)
        f_global = self.global_norm(f_global)
        
        f_cards = f_cards + f_global.unsqueeze(1)

        x_actions = x_actions.long()

        max_multi_select = (x_actions.shape[-1] - 9) // 2
        mo = max_multi_select * 2
        f_a_cards = self.get_action_card_(x_actions[..., :mo], f_cards)
        f_a_cards = f_a_cards + self.a_card_proj(self.a_card_norm(f_a_cards))

        x_a_feats = self.encode_action_(x_actions[..., mo:])
        x_a_feats = torch.cat(x_a_feats, dim=-1)
        f_actions = f_a_cards + self.a_feat_norm(x_a_feats)

        mask = x_actions[:, :, mo] == 0  # msg == 0
        valid = x['global_'][:, -1] == 0
        mask[:, 0] &= valid
        for layer in self.action_card_net:
            f_actions = layer(f_actions, f_cards, tgt_key_padding_mask=mask)

        if self.num_history_action_layers != 0:
            x_h_actions = x['h_actions_']
            x_h_actions = x_h_actions.long()

            x_h_id = self.get_h_action_card_(x_h_actions[..., :mo])

            x_h_a_feats = self.encode_action_(x_h_actions[:, :, mo:])
            x_h_a_feats = torch.cat(x_h_a_feats, dim=-1)
            f_h_actions = self.h_id_norm(x_h_id) + self.h_a_feat_norm(x_h_a_feats)
            
            for layer in self.action_history_net:
                f_actions = layer(f_actions, f_h_actions)

        f_actions = self.action_norm(f_actions)

        f_s_cards_global = f_cards.mean(dim=1)
        c_mask = 1 - mask.unsqueeze(-1).float()
        f_s_actions_ha = (f_actions * c_mask).sum(dim=1) / c_mask.sum(dim=1)
        f_state = torch.cat([f_s_cards_global, f_s_actions_ha], dim=-1)
        return f_actions, f_state, mask, valid


class Actor(nn.Module):

    def __init__(self, channels, use_transformer=False):
        super(Actor, self).__init__()
        c = channels
        self.state_proj = nn.Sequential(
            nn.Linear(c * 2, c),
            nn.ReLU(),
            nn.Linear(c, c),
        )
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.TransformerEncoderLayer(
                c, 4, c * 4, dropout=0.0, batch_first=True, norm_first=True, bias=True)
        self.head = nn.Sequential(
            nn.Linear(c, c // 4),
            nn.ReLU(),
            nn.Linear(c // 4, 1),
        )

    def forward(self, f_actions, h_state, mask):
        f_state = self.state_proj(h_state)
        # TODO: maybe token concat
        f_actions = f_actions + f_state.unsqueeze(1)
        if self.use_transformer:
            f_actions = self.transformer(f_actions, src_key_padding_mask=mask)
        logits = self.head(f_actions)[..., 0]
        logits = logits.float()
        logits = logits.masked_fill(mask, float("-inf"))
        return logits


class PPOAgent(nn.Module):

    def __init__(self, channels=128, num_card_layers=2, num_action_layers=2,
                 num_history_action_layers=2, embedding_shape=None, bias=False,
                 affine=True, a_trans=True, num_lstm_layers=1):
        super(PPOAgent, self).__init__()

        self.encoder = Encoder(
            channels, num_card_layers, num_action_layers, num_history_action_layers, embedding_shape, bias, affine)

        c = channels
        self.lstm = nn.LSTM(c * 2, c * 2, num_lstm_layers)

        self.actor = Actor(c, a_trans)

        self.critic = nn.Sequential(
            nn.Linear(c * 2, c // 2),
            nn.ReLU(),
            nn.Linear(c // 2, 1),
        )

        self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def load_embeddings(self, embeddings):
        self.encoder.load_embeddings(embeddings)
    
    def freeze_embeddings(self):
        self.encoder.freeze_embeddings()

    # def get_logit(self, x):
    #     f_actions, f_state, mask, valid = self.encoder(x)
    #     return self.actor(f_actions, mask)

    # def get_value(self, x):
    #     f_actions, f_state, mask, valid = self.encoder(x)
    #     return self.critic(f_state)

    def encode_lstm(self, hidden, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        new_hidden, lstm_state = self.lstm(hidden, lstm_state)
        # not_done = (~done.reshape((-1, batch_size))).float()
        # new_hidden = []
        # for i in range(hidden.shape[0]):
        #     h, lstm_state = self.lstm(
        #         hidden[i].unsqueeze(0),
        #         (
        #             not_done[i].view(1, -1, 1) * lstm_state[0],
        #             not_done[i].view(1, -1, 1) * lstm_state[1],
        #         ),
        #     )
        #     new_hidden += [h]
        # new_hidden = torch.cat(new_hidden)
        new_hidden = torch.flatten(new_hidden, 0, 1)
        return new_hidden, lstm_state

    def forward(self, x, lstm_state, done):
        f_actions, f_state, mask, valid = self.encoder(x)
        h_state, lstm_state = self.encode_lstm(f_state, lstm_state, done)
        logits = self.actor(f_actions, h_state, mask)
        return logits, self.critic(h_state), valid, lstm_state
