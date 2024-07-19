from enum import Enum
from itertools import combinations
import time

from typing import List

from .models import *
from pydantic import BaseModel, Field

import numpy as np

def sum_to2(w, ind, r):
    return sum_to2_helper(w, ind, 0, r)

def sum_to2_helper(w, ind, i, r):
    if r <= 0:
        return False
    n = len(ind)
    w_ = w[ind[i]]
    if i == n - 1:
        if len(w_) == 1:
            return w_[0] == r
        else:
            return w_[0] == r or w_[1] == r
    if len(w_) == 1:
        return sum_to2_helper(w, ind, i + 1, r - w_[0])
    else:
        return sum_to2_helper(w, ind, i + 1, r - w_[0]) or sum_to2_helper(w, ind, i + 1, r - w_[1])

def combinations_with_weight2(weights, r):
    n = len(weights)
    results = []

    for k in range(1, n + 1):
        combs = list(combinations(range(n), k))
        for comb in combs:
            if sum_to2(weights, comb, r):
                results.append(set(comb))
    return results

N_CARD_FEATURES = 41
MAX_CARDS = 80
MAX_ACTIONS = 24
N_ACTION_FEATURES = 12
N_GLOBAL_FEATURES = 23
N_HISTORY_ACTIONS = 32
H_ACTIONS_FEATS = 14
N_RNN_CHANNELS = 512

H_ACTIONS_SHAPE = (N_HISTORY_ACTIONS, H_ACTIONS_FEATS)
DESCRIPTION_LIMIT = 10000
CARD_EFFECT_OFFSET = 10010


def sample_input():
    history_actions = np.zeros(H_ACTIONS_SHAPE, dtype=np.uint8)
    cards = np.zeros((2*MAX_CARDS, N_CARD_FEATURES), dtype=np.uint8)
    global_ = np.zeros(N_GLOBAL_FEATURES, dtype=np.uint8)
    legal_actions = np.zeros((MAX_ACTIONS, N_ACTION_FEATURES), dtype=np.uint8)
    return {
        "cards_": cards,
        "global_": global_,
        "actions_": legal_actions,
        "h_actions_": history_actions,
    }

def init_rstate():
    return (
        np.zeros((1, N_RNN_CHANNELS), dtype=np.float32),
        np.zeros((1, N_RNN_CHANNELS), dtype=np.float32),
    )

system_strings = [
    1050, 1051, 1052, 1054, 1055, 1056, 1057, 1058, 1059, 1060,
    1061, 1062, 1063, 1064, 1066, 1067, 1068, 1069, 1070, 1071,
    1072, 1073, 1074, 1075, 1076, 1080, 1081, 1150, 1151, 1152,
    1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162,
    1163, 1164, 1165, 1166, 1167, 1168, 1169, 1190, 1191, 1192,
    1193, 1,    30,   31,   80,   81,   90,   91,   92,   93,
    94,   95,   96,   97,   98,   200,  203,  210,  218,  219,
    220,  221,  222,  221,  1621, 1622,
]

system_string_to_id = {
    s: i + 16 for i, s in enumerate(system_strings)
}

code_to_id = {}

def init_code_list(code_list_file):
    i = 1
    if len(code_to_id) == 0:
        code_to_id[0] = 0
        with open(code_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                code_to_id[int(line)] = i
                i += 1


def get_code_id(code):
    if code not in code_to_id:
        raise ValueError(f"Invalid code: {code}")
    return code_to_id[code]


location_to_id = {
    Location.deck: 1,
    Location.hand: 2,
    Location.mzone: 3,
    Location.szone: 4,
    Location.grave: 5,
    Location.removed: 6,
    Location.extra: 7,
}

controller_to_id = {
    Controller.me: 0,
    Controller.opponent: 1,
}

position_to_id = {
    Position.none: 0,
    Position.faceup_attack: 1,
    Position.facedown_attack: 2,
    Position.attack: 3,
    Position.faceup_defense: 4,
    Position.faceup: 5,
    Position.facedown_defense: 6,
    Position.facedown: 7,
    Position.defense: 8,
}

overlay_to_id = {
    True: 1,
    False: 0,
}

attribute_to_id = {
    Attribute.none: 0,
    Attribute.earth: 1,
    Attribute.water: 2,
    Attribute.fire: 3,
    Attribute.wind: 4,
    Attribute.light: 5,
    Attribute.dark: 6,
    Attribute.divine: 7,
}

race_to_id = {
    Race.none: 0,
    Race.warrior: 1,
    Race.spellcaster: 2,
    Race.fairy: 3,
    Race.fiend: 4,
    Race.zombie: 5,
    Race.machine: 6,
    Race.aqua: 7,
    Race.pyro: 8,
    Race.rock: 9,
    Race.windbeast: 10,
    Race.plant: 11,
    Race.insect: 12,
    Race.thunder: 13,
    Race.dragon: 14,
    Race.beast: 15,
    Race.beast_warrior: 16,
    Race.dinosaur: 17,
    Race.fish: 18,
    Race.sea_serpent: 19,
    Race.reptile: 20,
    Race.psycho: 21,
    Race.devine: 22,
    Race.creator_god: 23,
    Race.wyrm: 24,
    Race.cyberse: 25,
    Race.illusion: 26,
}

negated_to_id = {
    True: 1,
    False: 0,
}

type_to_id = {
    Type.monster: 0,
    Type.spell: 1,
    Type.trap: 2,
    Type.normal: 3,
    Type.effect: 4,
    Type.fusion: 5,
    Type.ritual: 6,
    Type.trap_monster: 7,
    Type.spirit: 8,
    Type.union: 9,
    Type.dual: 10,
    Type.tuner: 11,
    Type.synchro: 12,
    Type.token: 13,
    Type.quick_play: 14,
    Type.continuous: 15,
    Type.equip: 16,
    Type.field: 17,
    Type.counter: 18,
    Type.flip: 19,
    Type.toon: 20,
    Type.xyz: 21,
    Type.pendulum: 22,
    Type.special: 23,
    Type.link: 24,
}

phase_to_id = {
    Phase.draw: 0,
    Phase.standby: 1,
    Phase.main1: 2,
    Phase.battle_start: 3,
    Phase.battle_step: 4,
    Phase.damage: 5,
    Phase.damage_calculation: 6,
    Phase.battle: 7,
    Phase.main2: 8,
    Phase.end: 9,
}

msg_to_id = {
    MsgName.select_idlecmd: 1,
    MsgName.select_chain: 2,
    MsgName.select_card: 3,
    MsgName.select_tribute: 4,
    MsgName.select_position: 5,
    MsgName.select_effectyn: 6,
    MsgName.select_yesno: 7,
    MsgName.select_battlecmd: 8,
    MsgName.select_unselect_card: 9,
    MsgName.select_option: 10,
    MsgName.select_place: 11,
    MsgName.select_sum: 12,
    MsgName.select_disfield: 13,
    MsgName.announce_attrib: 14,
    MsgName.announce_number: 15,
}


class ActionAct(Enum):
    none = 'none'
    set = 'set'
    reposition = 'reposition'
    special_summon = 'special_summon'
    summon_faceup_attack = 'summon_faceup_attack'
    summon_facedown_defense = 'summon_facedown_defense'
    attack = 'attack'
    direct_attack = 'direct_attack'
    activate = 'activate'
    cancel = 'cancel'


action_act_to_id = {
    ActionAct.none: 0,
    ActionAct.set: 1,
    ActionAct.reposition: 2,
    ActionAct.special_summon: 3,
    ActionAct.summon_faceup_attack: 4,
    ActionAct.summon_facedown_defense: 5,
    ActionAct.attack: 6,
    ActionAct.direct_attack: 7,
    ActionAct.activate: 8,
    ActionAct.cancel: 9,
}


class ActionPhase(Enum):
    none = 'none'
    battle = 'battle'
    main2 = 'main2'
    end = 'end'


action_phase_to_id = {
    ActionPhase.none: 0,
    ActionPhase.battle: 1,
    ActionPhase.main2: 2,
    ActionPhase.end: 3,
}


class ActionPlace(Enum):
    none = 'none'
    m1 = 'm1'
    m2 = 'm2'
    m3 = 'm3'
    m4 = 'm4'
    m5 = 'm5'
    m6 = 'm6'
    m7 = 'm7'
    s1 = 's1'
    s2 = 's2'
    s3 = 's3'
    s4 = 's4'
    s5 = 's5'
    s6 = 's6'
    s7 = 's7'
    s8 = 's8'
    om1 = 'om1'
    om2 = 'om2'
    om3 = 'om3'
    om4 = 'om4'
    om5 = 'om5'
    om6 = 'om6'
    om7 = 'om7'
    os1 = 'os1'
    os2 = 'os2'
    os3 = 'os3'
    os4 = 'os4'
    os5 = 'os5'
    os6 = 'os6'
    os7 = 'os7'
    os8 = 'os8'


def place_to_select(place: Place) -> ActionPlace:
    s = ""
    if place.controller == Controller.opponent:
        s += "o"
    if place.location == Location.mzone:
        s += "m"
    elif place.location == Location.szone:
        s += "s"
    else:
        raise ValueError(f"Invalid location: {place.location}")
    s += str(place.sequence + 1)
    return ActionPlace[s]


place_to_id = {
    ActionPlace.none: 0,
    ActionPlace.m1: 1,
    ActionPlace.m2: 2,
    ActionPlace.m3: 3,
    ActionPlace.m4: 4,
    ActionPlace.m5: 5,
    ActionPlace.m6: 6,
    ActionPlace.m7: 7,
    ActionPlace.s1: 8,
    ActionPlace.s2: 9,
    ActionPlace.s3: 10,
    ActionPlace.s4: 11,
    ActionPlace.s5: 12,
    ActionPlace.s6: 13,
    ActionPlace.s7: 14,
    ActionPlace.s8: 15,
    ActionPlace.om1: 16,
    ActionPlace.om2: 17,
    ActionPlace.om3: 18,
    ActionPlace.om4: 19,
    ActionPlace.om5: 20,
    ActionPlace.om6: 21,
    ActionPlace.om7: 22,
    ActionPlace.os1: 23,
    ActionPlace.os2: 24,
    ActionPlace.os3: 25,
    ActionPlace.os4: 26,
    ActionPlace.os5: 27,
    ActionPlace.os6: 28,
    ActionPlace.os7: 29,
    ActionPlace.os8: 30,
}


class LegalAction(BaseModel):
    msg: MsgName = Field(..., examples=['select_chain'])
    spec: str = Field(
        "",
        description='The card spec, e.g. "m1", "s2a1"'
    )
    act: ActionAct = Field(
        ActionAct.none,
        description='Legal in select_idlecmd, select_chain and select_battlecmd',
        examples=['activate'],
    )
    phase: ActionPhase = Field(
        ActionPhase.none,
        description='Legal in select_idlecmd and select_battlecmd'
    )
    finish: bool = Field(
        False,
        description='Legal in select_card, select_tribute, select_sum and select_unselect_card'
    )
    position: Position = Field(
        Position.none, description='Legal in select_position')
    effect: int = Field(
        -1,
        description='effect index'
    )
    number: int = Field(
        0, description='Legal in announce_number. 0 is N/A'
    )
    place: ActionPlace = Field(
        ActionPlace.none,
        description='Legal in select_place and select_disfield'
    )
    attribute: Attribute = Field(
        Attribute.none,
        description='Legal in announce_attrib'
    )

    card_index: int = Field(
        0, description='The the array index of the card in cards. 0 is N/A or unknown.'
    )
    card_id: int = Field(
        0, description='The card id.')
    response: int = Field(
        -100, description="The response to be send to the server.")
    can_finish: bool = Field(
        False, description='Temporary field for select_sum')


def to_spec(x: Union[CardInfo, CardLocation]):
    if isinstance(x, CardInfo):
        x = CardLocation(
            controller=x.controller,
            location=x.location,
            sequence=x.sequence,
            overlay_sequence=-1,
        )
    spec = ""
    if x.location == Location.hand:
        spec += 'h'
    elif x.location == Location.mzone:
        spec += 'm'
    elif x.location == Location.szone:
        spec += 's'
    elif x.location == Location.grave:
        spec += 'g'
    elif x.location == Location.removed:
        spec += 'r'
    elif x.location == Location.extra:
        spec += 'x'
    elif x.location == Location.deck:
        pass
    else:
        raise ValueError(f"Unknown location: {x.location}")
    spec += str(x.sequence + 1)
    if x.overlay_sequence >= 0:
        spec += f"a{x.overlay_sequence + 1}"
    if x.controller == Controller.opponent:
        spec = 'o' + spec
    return spec


def int_transform(x):
    return x // 256, x % 256


def float_transform(x):
    x = int(x) % 65536
    return x // 256, x % 256


def encode_card(card: Card):
    x = np.zeros(N_CARD_FEATURES, dtype=np.uint8)

    x[0:2] = int_transform(get_code_id(card.code))

    x[2] = location_to_id[card.location]
    if card.location in [Location.mzone, Location.szone, Location.grave]:
        x[3] = card.sequence + 1
    x[4] = controller_to_id[card.controller]

    position = card.position
    overlay = card.overlay_sequence != -1
    if overlay:
        position = Position.faceup
    elif card.location in [Location.deck, Location.hand, Location.extra]:
        if position in [Position.facedown_defense, Position.facedown, Position.facedown_attack]:
            position = Position.facedown
    x[5] = position_to_id[position]

    x[6] = overlay_to_id[overlay]
    x[7] = attribute_to_id[card.attribute]
    x[8] = race_to_id[card.race]
    x[9] = min(max(card.level, 0), 13)
    x[10] = min(max(card.counter, 0), 15)
    x[11] = negated_to_id[card.negated]

    x[12:14] = float_transform(card.attack)

    x[14:16] = float_transform(card.defense)

    for c in card.types:
        x[16 + type_to_id[c]] = 1
    return x


def get_spec(c: Card):
    return to_spec(CardLocation(
        controller=c.controller,
        location=c.location,
        sequence=c.sequence,
        overlay_sequence=c.overlay_sequence,
    ))


def encode_cards(cards: List[Card]):
    spec_infos = {}
    cards = cards[:2*MAX_CARDS]
    x = np.zeros((2*MAX_CARDS, N_CARD_FEATURES), dtype=np.uint8)
    for i, card in enumerate(cards):
        x[i] = encode_card(card)
        spec = get_spec(card)
        spec_infos[spec] = (i + 1, get_code_id(card.code))
    return x, spec_infos


def count_location_cards(cards: List[Card]):
    n_my_decks = 0
    n_my_hands = 0
    n_my_mzones = 0
    n_my_szones = 0
    n_my_graves = 0
    n_my_removeds = 0
    n_my_extras = 0
    n_op_decks = 0
    n_op_hands = 0
    n_op_mzones = 0
    n_op_szones = 0
    n_op_graves = 0
    n_op_removeds = 0
    n_op_extras = 0
    for c in cards:
        if c.controller == Controller.me:
            if c.location == Location.deck:
                n_my_decks += 1
            elif c.location == Location.hand:
                n_my_hands += 1
            elif c.location == Location.mzone:
                n_my_mzones += 1
            elif c.location == Location.szone:
                n_my_szones += 1
            elif c.location == Location.grave:
                n_my_graves += 1
            elif c.location == Location.removed:
                n_my_removeds += 1
            elif c.location == Location.extra:
                n_my_extras += 1
        elif c.controller == Controller.opponent:
            if c.location == Location.deck:
                n_op_decks += 1
            elif c.location == Location.hand:
                n_op_hands += 1
            elif c.location == Location.mzone:
                n_op_mzones += 1
            elif c.location == Location.szone:
                n_op_szones += 1
            elif c.location == Location.grave:
                n_op_graves += 1
            elif c.location == Location.removed:
                n_op_removeds += 1
            elif c.location == Location.extra:
                n_op_extras += 1
    return n_my_decks, n_my_hands, n_my_mzones, n_my_szones, n_my_graves, n_my_removeds, n_my_extras, \
        n_op_decks, n_op_hands, n_op_mzones, n_op_szones, n_op_graves, n_op_removeds, n_op_extras


def encode_global(g: Global, cards: List[Card]):
    x = np.zeros(N_GLOBAL_FEATURES, dtype=np.uint8)
    x[0:2] = float_transform(g.my_lp)
    x[2:4] = float_transform(g.op_lp)
    x[4] = min(max(g.turn, 0), 16)
    x[5] = phase_to_id[g.phase]
    x[6] = int(g.is_first)
    x[7] = int(g.is_my_turn)
    x[8:22] = count_location_cards(cards)
    # x[22] = 0
    return x


def encode_action(action: LegalAction):
    x = np.zeros(N_ACTION_FEATURES, dtype=np.uint8)
    x[0] = action.card_index
    x[1:3] = int_transform(action.card_id)
    x[3] = msg_to_id[action.msg]
    x[4] = action_act_to_id[action.act]
    x[5] = 1 if action.finish else 0

    effect = action.effect
    if effect == -1:
        effect = 0
    elif effect == 0:
        effect = 1
    elif effect >= CARD_EFFECT_OFFSET:
        effect = effect - CARD_EFFECT_OFFSET + 2
    else:
        effect = system_string_to_id[effect]
    x[6] = effect

    x[7] = action_phase_to_id[action.phase]
    x[8] = position_to_id[action.position]
    x[9] = min(max(action.number, 0), 12)
    x[10] = place_to_id[action.place]
    x[11] = attribute_to_id[action.attribute]
    return x


def find_spec_info(spec_infos, spec):
    if spec in spec_infos:
        return spec_infos[spec]
    return 0, 0


def unpack_desc(code, desc):
    if desc < DESCRIPTION_LIMIT:
        return 0, desc
    code_ = desc >> 4
    idx = desc & 0xf
    if idx < 0 or idx >= 14:
        print(f"Code: {code}, Code_: {code_}, Desc: {desc}")
        raise ValueError(f"Invalid effect index: {idx}")
    return code_, idx + CARD_EFFECT_OFFSET

def posotion_to_response(position):
    if position == Position.faceup_attack:
        return 0x1
    elif position == Position.facedown_attack:
        return 0x2
    elif position == Position.faceup_defense:
        return 0x4
    elif position == Position.facedown_defense:
        return 0x8
    raise ValueError(f"Invalid position: {position}")

# ActionMsg (len1)
#    get_legal_actions: skip some actions
# -> List[LegalAction] (len2)
# -> np.ndarray (len2)
#    truncate or pad to MAX_ACTIONS
# -> inputs (len3)
# -> outputs (len3)
#    revert truncation or padding
# -> probs (len2)
#    add skipped actions back
# -> return_probs (len1)

def get_legal_actions(action_msg: ActionMsg) -> List[LegalAction]:
    if action_msg.data.msg_type == "select_idlecmd":
        actions = []
        msg: MsgSelectIdleCmd = action_msg.data
        for cmd in msg.idle_cmds:
            action = LegalAction(msg=MsgName.select_idlecmd)
            if cmd.data is not None:
                action.response = cmd.data.response
                action.spec = to_spec(cmd.data.card_info)
            if cmd.cmd_type == IdleCmdType.summon:
                action.act = ActionAct.summon_faceup_attack
            elif cmd.cmd_type == IdleCmdType.sp_summon:
                action.act = ActionAct.special_summon
            elif cmd.cmd_type == IdleCmdType.reposition:
                action.act = ActionAct.reposition
            elif cmd.cmd_type == IdleCmdType.mset:
                action.act = ActionAct.summon_facedown_defense
            elif cmd.cmd_type == IdleCmdType.set:
                action.act = ActionAct.set
            elif cmd.cmd_type == IdleCmdType.activate:
                desc = cmd.data.effect_description
                code = cmd.data.card_info.code
                if code & 0x80000000:
                    code &= 0x7fffffff
                code_d, eff_idx = unpack_desc(code, desc)
                if desc == 0:
                    code_d = code
                action.act = ActionAct.activate
                action.spec = to_spec(cmd.data.card_info)
                action.effect = eff_idx
                if code_d != 0:
                    action.card_id = get_code_id(code_d)
            elif cmd.cmd_type == IdleCmdType.to_bp:
                action.phase = ActionPhase.battle
                action.response = 6
            elif cmd.cmd_type == IdleCmdType.to_ep:
                # TODO: Train model to support it
                # if has_bp:
                #     continue
                action.phase = ActionPhase.end
                action.response = 7
            actions.append(action)
    elif action_msg.data.msg_type == "select_chain":
        actions = []
        msg: MsgSelectChain = action_msg.data
        for i, chain in enumerate(msg.chains):
            action = LegalAction(msg=MsgName.select_chain)
            action.response = chain.response
            code = chain.code
            desc = chain.effect_description
            code_d, eff_idx = unpack_desc(code, desc)
            if desc == 0:
                code_d = code
            action.act = ActionAct.activate
            action.spec = to_spec(chain.location)
            action.effect = eff_idx
            if code_d != 0:
                action.card_id = get_code_id(code_d)
            actions.append(action)
        if not msg.forced:
            action = LegalAction(msg=MsgName.select_chain)
            action.response = -1
            action.act = ActionAct.cancel
            actions.append(action)
    elif action_msg.data.msg_type == "select_position":
        actions = []
        msg: MsgSelectPosition = action_msg.data
        for pos in msg.positions:
            action = LegalAction(msg=MsgName.select_position)
            action.position = pos
            action.response = posotion_to_response(pos)
            actions.append(action)
    elif action_msg.data.msg_type == "select_yesno":
        actions = []
        msg: MsgSelectYesNo = action_msg.data
        desc = msg.effect_description
        code, eff_idx = unpack_desc(0, desc)
        if desc == 0:
            raise ValueError(f"Unknown desc {desc} in select_yesno")
        action = LegalAction(msg=MsgName.select_yesno)
        action.response = 1
        action.act = ActionAct.activate
        action.effect = eff_idx
        if code != 0:
            action.card_id = get_code_id(code)
        actions.append(action)

        action = LegalAction(msg=MsgName.select_yesno)
        action.response = 0
        action.act = ActionAct.cancel
        actions.append(action)
    elif action_msg.data.msg_type == "select_effectyn":
        actions = []
        msg: MsgSelectEffectYn = action_msg.data

        action = LegalAction(msg=MsgName.select_effectyn)
        action.response = 1
        code = msg.code
        desc = msg.effect_description
        code_d, eff_idx = unpack_desc(code, desc)
        if desc == 0:
            code_d = code
        action.act = ActionAct.activate
        action.spec = to_spec(msg.location)
        action.effect = eff_idx
        if code_d != 0:
            action.card_id = get_code_id(code_d)
        actions.append(action)

        action = LegalAction(msg=MsgName.select_effectyn)
        action.response = 0
        action.act = ActionAct.cancel
        actions.append(action)
    elif action_msg.data.msg_type == "select_battlecmd":
        actions = []
        msg: MsgSelectBattleCmd = action_msg.data
        for cmd in msg.battle_cmds:
            action = LegalAction(msg=MsgName.select_battlecmd)
            if cmd.data is not None:
                action.spec = to_spec(cmd.data.card_info)
                action.response = cmd.data.response
            if cmd.cmd_type == BattleCmdType.activate:
                action.act = ActionAct.activate
                code_t = cmd.data.card_info.code
                desc = cmd.data.effect_description
                code = code_t
                if code_t & 0x80000000:
                    code_t &= 0x7fffffff
                code_d, eff_idx = unpack_desc(code_t, desc)
                if desc == 0:
                    code_d = code
                action.effect = eff_idx
                if code_d != 0:
                    action.card_id = get_code_id(code_d)
            elif cmd.cmd_type == BattleCmdType.attack:
                if cmd.data.direct_attackable:
                    action.act = ActionAct.direct_attack
                else:
                    action.act = ActionAct.attack
            elif cmd.cmd_type == BattleCmdType.to_m2:
                action.phase = ActionPhase.main2
                action.response = 2
            elif cmd.cmd_type == BattleCmdType.to_ep:
                # TODO: Train model to support it
                # if has_m2:
                #     continue
                action.response = 3
                action.phase = ActionPhase.end
            actions.append(action)
    elif action_msg.data.msg_type == "select_option":
        actions = []
        msg: MsgSelectOption = action_msg.data
        for option in msg.options:
            desc = option.code
            code, eff_idx = unpack_desc(0, desc)
            if desc == 0:
                raise ValueError(f"Unknown desc {desc} in select_option")
            action = LegalAction(msg=MsgName.select_option)
            action.response = option.response
            action.act = ActionAct.activate
            action.effect = eff_idx
            if code != 0:
                action.card_id = get_code_id(code)
            actions.append(action)
    elif action_msg.data.msg_type == "select_place":
        actions = []
        msg: MsgSelectPlace = action_msg.data
        for i, place in enumerate(msg.places):
            action = LegalAction(msg=MsgName.select_place)
            action.response = i
            action.place = place_to_select(place)
            actions.append(action)
    elif action_msg.data.msg_type == "select_disfield":
        actions = []
        msg: MsgSelectDisfield = action_msg.data
        for place in msg.places:
            action = LegalAction(msg=MsgName.select_disfield)
            action.response = -1
            action.place = place_to_select(place)
            actions.append(action)
    elif action_msg.data.msg_type == "announce_attrib":
        actions = []
        msg: MsgAnnounceAttrib = action_msg.data
        if msg.count != 1:
            raise NotImplementedError("Multiple attributes are not supported.")
        for attrib in msg.attributes:
            action = LegalAction(msg=MsgName.announce_attrib)
            action.response = attrib.response
            action.attribute = attrib.attribute
            actions.append(action)
    elif action_msg.data.msg_type == "announce_number":
        actions = []
        msg: MsgAnnounceNumber = action_msg.data
        if msg.count != 1:
            raise NotImplementedError("Multiple numbers are not supported.")
        for number in msg.numbers:
            if number <= 0 or number > 12:
                raise NotImplementedError(
                    "Number out of range, only 1-12 are supported.")
            action = LegalAction(msg=MsgName.announce_number)
            action.response = number.response
            action.number = number.number
            actions.append(action)
    elif action_msg.data.msg_type == "select_unselect_card":
        actions = []
        msg: MsgSelectUnselectCard = action_msg.data
        for card in msg.selectable_cards:
            action = LegalAction(msg=MsgName.select_unselect_card)
            action.response = card.response
            action.spec = to_spec(card.location)
            actions.append(action)
        if msg.finishable:
            action = LegalAction(msg=MsgName.select_unselect_card)
            action.response = -1
            action.finish = True
            actions.append(action)
    elif action_msg.data.msg_type in ["select_card", "select_tribute"]:
        actions = []
        if action_msg.data.msg_type == "select_card":
            msg: MsgSelectCard = action_msg.data
            if msg.min == 0:
                raise NotImplementedError("min=0 is not supported.")
            msg_name = MsgName.select_card
        else:
            msg: MsgSelectTribute = action_msg.data
            if msg.min == 0:
                raise NotImplementedError(
                    "min=0 is not supported for select_tribute.")
            if msg.min != msg.max:
                raise NotImplementedError(
                    "min != max is not supported for select_tribute.")
            if any(c.level != 1 for c in msg.cards):
                raise NotImplementedError(
                    "Only level=1 cards are supported for select_tribute.")
            msg_name = MsgName.select_tribute
        specs = [to_spec(c.location) for c in msg.cards]
        responses = [c.response for c in msg.cards]
        idx = len(msg.selected)
        for i, spec in enumerate(specs):
            if i not in msg.selected:
                action = LegalAction(msg=msg_name)
                action.response = responses[i]
                action.spec = spec
                actions.append(action)
        if (idx == msg.max - 1 and idx >= msg.min) or idx >= msg.min:
            action = LegalAction(msg=msg_name)
            action.response = -1
            action.finish = True
            actions.append(action)
    elif action_msg.data.msg_type == "select_sum":
        msg: MsgSelectSum = action_msg.data
        if msg.overflow:
            raise NotImplementedError(
                "overflow is not supported for select_sum.")
        elif len(msg.must_cards) > 2:
            raise NotImplementedError(
                "must select more than 2 cards is not supported for select_sum.")
        level_sum = msg.level_sum
        for c in msg.must_cards:
            level_sum -= c.level1
        card_levels = []
        for i, c in enumerate(msg.cards):
            levels = []
            if c.level1 > 0:
                levels.append(c.level1)
            if c.level2 > 0 and c.level2 != c.level1:
                levels.append(c.level2)
            card_levels.append(levels)
        # Generate all possible combinations
        combs = combinations_with_weight2(card_levels, level_sum)
        # find combinations contains selected
        selected = set(msg.selected)
        combs = [
            c - selected for c in combs if c.intersection(selected) == selected
        ]
        # deduplicate
        combs2 = []
        for c in combs:
            if c not in combs2:
                combs2.append(c)
        if set() in combs2:
            raise NotImplementedError("empty select in select_sum.")
        can_finish = {}
        for c in combs2:
            i = next(iter(c))
            f = len(c) == 1
            if i not in can_finish:
                can_finish[i] = f
            else:
                can_finish[i] = can_finish[i] or f
        actions = []
        for i, f in can_finish.items():
            action = LegalAction(msg=MsgName.select_tribute)
            c = msg.cards[i]
            action.response = c.response
            action.spec = to_spec(c.location)
            action.can_finish = f
            actions.append(action)
    return actions


def encode_legal_actions(actions: List[LegalAction], spec_infos):
    print(actions[0].msg)
    actions = actions[:MAX_ACTIONS]
    x = np.zeros((MAX_ACTIONS, N_ACTION_FEATURES), dtype=np.uint8)
    for i, action in enumerate(actions):
        card_index, card_id = find_spec_info(spec_infos, action.spec)
        action.card_index = card_index
        if action.card_id == 0:
            action.card_id = card_id
        x[i] = encode_action(action)
    return x


def transform_select_idx(probs, idx: int, action_msg: ActionMsg):
    if probs[idx] == -1:
        raise ValueError("Invalid action selected (prob == -1)")
    k = idx
    if action_msg.data.msg_type in ["select_card", "select_tribute", "select_sum"]:
        k = 0
        for i in range(len(probs)):
            if probs[i] != -1:
                if i == idx:
                    break
                k += 1
    # If idx is pad, it is input error, we don't need to handle it
    # If idx is truncated, it is -1 and already handled
    return k


def add_skipped_back(probs, legal_actions, action_msg: ActionMsg):
    # TODO: Train model to support it
    # if action_msg.name == MsgName.select_idlecmd:
    #     msg: MsgSelectIdleCmd = action_msg.data
    #     bp_idx = -1
    #     ep_idx = -1
    #     for i, cmd in enumerate(msg.idle_cmds):
    #         if cmd.cmd_type == IdleCmdType.to_bp:
    #             bp_idx = i
    #         elif cmd.cmd_type == IdleCmdType.to_ep:
    #             ep_idx = i
    #     if bp_idx >= 0 and ep_idx >= 0:
    #         probs.insert(ep_idx, 0)
    # elif action_msg.name == MsgName.select_battlecmd:
    #     msg: MsgSelectBattleCmd = action_msg.data
    #     m2_idx = -1
    #     ep_idx = -1
    #     for i, cmd in enumerate(msg.battle_cmds):
    #         if cmd.cmd_type == BattleCmdType.to_m2:
    #             m2_idx = i
    #         elif cmd.cmd_type == BattleCmdType.to_ep:
    #             ep_idx = i
    #     if m2_idx >= 0 and ep_idx >= 0:
    #         probs.insert(ep_idx, 0)
    msg_type = action_msg.data.msg_type
    responses = [a.response for a in legal_actions]
    can_finish = [False] * len(responses)
    if msg_type in ["select_card", "select_tribute", "select_sum"]:
        can_finish = [a.can_finish for a in legal_actions]
        if msg_type == 'select_sum':
            skipped = [
                i for i, c in enumerate(action_msg.data.cards)
                if c.response not in responses
            ]
            for i in skipped:
                probs.insert(i, -1)
                responses.insert(i, action_msg.data.cards[i].response)
                can_finish.insert(i, False)
        else:
            skipped = action_msg.data.selected
            for i in skipped:
                probs.insert(i, -1)
                responses.insert(i, action_msg.data.cards[i].response)
            if len(probs) == len(action_msg.data.cards):
                # finish
                probs.append(-1)
                responses.append(-1)
    return probs, responses, can_finish


def encode_history_actions(h_actions, ha_p, turn):
    x = np.zeros(H_ACTIONS_SHAPE, dtype=np.uint8)
    n = x.shape[0]
    n1 = n - ha_p
    x[:n1] = h_actions[ha_p:]
    x[n1:] = h_actions[:ha_p]
    turn_diff = np.minimum(16, turn - x[:, 12])
    x[:, 12] = np.where(x[:, 3] != 0, turn_diff, 0)
    return x


class HistoryActions:

    def __init__(self):
        self.h_actions = np.zeros(H_ACTIONS_SHAPE, dtype=np.uint8)
        self.ha_p = 0
    
    def encode(self, turn):
        return encode_history_actions(self.h_actions, self.ha_p, turn)

    def update(self, action: np.ndarray, turn: int, phase: Phase):
        self.ha_p -= 1
        if self.ha_p < 0:
            self.ha_p = self.h_actions.shape[0] - 1
        self.h_actions[self.ha_p, :action.shape[0]] = action
        self.h_actions[self.ha_p, 0] = 0
        self.h_actions[self.ha_p, 12] = turn
        self.h_actions[self.ha_p, 13] = phase_to_id[phase]


class PredictState:

    def __init__(self):
        self.rstate = init_rstate()
        self.index = 0
        self.history_actions = HistoryActions()

        self.reset()

        self._timestamp = time.time()

    def reset(self):
        self._probs = None
        self._actions = None
        self._action_msg = None
        self._turn = None
        self._phase = None
    
    def update_history_actions(self, idx: int):
        idx1 = transform_select_idx(self._probs, idx, self._action_msg)
        action = self._actions[idx1]
        self.history_actions.update(action, self._turn, self._phase)
        self.reset()

    def record(self, input: Input, actions, probs):
        self._probs = probs
        self._actions = actions
        self._action_msg = input.action_msg
        self._turn = input.global_.turn
        self._phase = input.global_.phase
        self._timestamp = time.time()

def revert_pad_truncate(probs, n_actions):
    if len(probs) < n_actions:
        probs += [-1] * (n_actions - len(probs))
    elif len(probs) > n_actions:
        probs = probs[:n_actions]
    return probs

def predict(model_fn, input: Input, prev_action_idx, state: PredictState):
    if state.index != 0:
        state.update_history_actions(prev_action_idx)

    legal_actions = get_legal_actions(input.action_msg)
    n_actions = len(legal_actions)

    cards, spec_infos = encode_cards(input.cards)
    global_ = encode_global(input.global_, input.cards)
    actions = encode_legal_actions(legal_actions, spec_infos)
    h_actions = state.history_actions.encode(input.global_.turn)
    model_input = {
        "cards_": cards,
        "global_": global_,
        "actions_": actions,
        "h_actions_": h_actions,
    }
    if n_actions == 1:
        probs = [1.0]
        responses = [legal_actions[0].response]
        win_rate = -1
        can_finish = [legal_actions[0].can_finish]
    else:
        rstate, probs, value = model_fn(state.rstate, model_input)
        state.rstate = rstate
        probs = revert_pad_truncate(probs, n_actions)
        assert len(probs) == n_actions
        probs, responses, can_finish = add_skipped_back(probs, legal_actions, input.action_msg)
        win_rate = (value + 1) / 2
    assert len(probs) == len(responses)
    preds = [
        ActionPredict(prob=prob, response=response, can_finish=f)
        for prob, response, f in zip(probs, responses, can_finish)
    ]
    predict_results = MsgResponse(
        action_preds=preds,
        win_rate=win_rate,
    )
    state.record(input, actions, probs)
    state.index += 1
    return predict_results


class Predictor:
    def __init__(self, loaded, predict_fn):
        self.loaded = loaded
        self.predict_fn = predict_fn
    
    def predict(self, rstate, sample_obs):
        return self.predict_fn(self.loaded, rstate, sample_obs)
    
    @staticmethod
    def load(checkpoint, num_threads):
        sample_obs = sample_input()
        rstate = init_rstate()
        if checkpoint.endswith(".flax_model"):
            from .jax_inf import load_model, predict_fn
        elif checkpoint.endswith(".tflite"):
            from .tflite_inf import load_model, predict_fn
        predictor = Predictor(load_model(checkpoint, rstate, sample_obs, num_threads=num_threads), predict_fn)
        predictor.predict(rstate, sample_obs)
        return predictor
