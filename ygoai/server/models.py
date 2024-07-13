from enum import Enum
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field


class ActionPredict(BaseModel):
    prob: float
    response: int
    can_finish: bool = Field(False, description='only used in select_sum')


class MsgResponse(BaseModel):
    action_preds: List[ActionPredict]
    win_rate: float


class AnnounceNumber(BaseModel):
    number: int
    response: int

class MsgAnnounceNumber(BaseModel):
    msg_type: Literal['announce_number']
    count: int = Field(..., description='!= 1 not supported')
    numbers: List[AnnounceNumber]


class Attribute(Enum):
    none = 'none'
    earth = 'earth'
    water = 'water'
    fire = 'fire'
    wind = 'wind'
    light = 'light'
    dark = 'dark'
    divine = 'divine'


class AnnounceAttrib(BaseModel):
    attribute: Attribute
    response: int

class MsgAnnounceAttrib(BaseModel):
    msg_type: Literal['announce_attrib']
    count: int = Field(..., description='!= 1 not supported')
    attributes: List[AnnounceAttrib]


class Controller(Enum):
    me = 'me'
    opponent = 'opponent'


class Location(Enum):
    deck = 'deck'
    hand = 'hand'
    mzone = 'mzone'
    szone = 'szone'
    grave = 'grave'
    removed = 'removed'
    extra = 'extra'


class Place(BaseModel):
    controller: Controller = Field(..., examples=['me'])
    location: Location = Field(..., examples=['hand', 'deck'])
    sequence: int = Field(..., description='Start from 0')


class MsgSelectDisfield(BaseModel):
    msg_type: Literal['select_disfield']
    count: int = Field(..., description='> 1 not supported; 0 is considered as 1.')
    places: List[Place]


class MsgSelectPlace(BaseModel):
    msg_type: Literal['select_place']
    count: int = Field(..., description='> 1 not supported; 0 is considered as 1.')
    places: List[Place]


class Option(BaseModel):
    code: int
    response: int


class MsgSelectOption(BaseModel):
    msg_type: Literal['select_option']
    options: List[Option] = Field(..., description='ignored')


class CardLocation(BaseModel):
    controller: Controller = Field(..., examples=['me'])
    location: Location = Field(..., examples=['hand', 'deck'])
    sequence: int = Field(..., description='Start from 0')
    overlay_sequence: int = Field(
        ...,
        description='if is overlay, this is the overlay index, starting from 0, else -1.',
    )

class CardInfo(BaseModel):
    code: int
    controller: Controller = Field(..., examples=['me'])
    location: Location = Field(..., examples=['hand', 'deck'])
    sequence: int = Field(..., description='Start from 0')


class MsgSelectYesNo(BaseModel):
    msg_type: Literal['select_yesno']
    effect_description: int


class MsgSelectEffectYn(BaseModel):
    msg_type: Literal['select_effectyn']
    code: int
    location: CardLocation
    effect_description: int


class Position(Enum):
    none = 'none'
    faceup_attack = 'faceup_attack'
    facedown_attack = 'facedown_attack'
    attack = 'attack'
    faceup_defense = 'faceup_defense'
    faceup = 'faceup'
    facedown_defense = 'facedown_defense'
    facedown = 'facedown'
    defense = 'defense'


class MsgSelectPosition(BaseModel):
    msg_type: Literal['select_position']
    code: int
    positions: List[Position]


class Chain(BaseModel):
    code: int
    location: CardLocation
    effect_description: int
    response: int


class MsgSelectChain(BaseModel):
    msg_type: Literal['select_chain']
    forced: bool
    chains: List[Chain]


class IdleCmdType(Enum):
    summon = 'summon'
    sp_summon = 'sp_summon'
    reposition = 'reposition'
    mset = 'mset'
    set = 'set'
    activate = 'activate'
    to_bp = 'to_bp'
    to_ep = 'to_ep'


class IdleCmdData(BaseModel):
    card_info: CardInfo
    effect_description: int
    response: int


class IdleCmd(BaseModel):
    cmd_type: IdleCmdType
    data: Optional[IdleCmdData] = None


class MsgSelectIdleCmd(BaseModel):
    msg_type: Literal['select_idlecmd']
    idle_cmds: List[IdleCmd]


class SelectSumCard(BaseModel):
    location: CardLocation
    level1: int
    level2: int
    response: int


class MsgSelectSum(BaseModel):
    msg_type: Literal['select_sum']
    overflow: bool = Field(..., description='true not supported')
    level_sum: int
    min: int
    max: int
    cards: List[SelectSumCard]
    must_cards: List[SelectSumCard] = Field(..., description='size > 2 not supported')
    selected: List[int]


class SelectTributeCard(BaseModel):
    location: CardLocation
    level: int
    response: int


class MsgSelectTribute(BaseModel):
    msg_type: Literal['select_tribute']
    cancelable: bool = Field(..., description='ignored')
    min: int
    max: int
    cards: List[SelectTributeCard]
    selected: List[int]


class SelectAbleCard(BaseModel):
    location: CardLocation
    response: int


class MsgSelectCard(BaseModel):
    msg_type: Literal['select_card']
    cancelable: bool = Field(..., description='ignored')
    min: int
    max: int
    cards: List[SelectAbleCard]
    selected: List[int]


class Race(Enum):
    none = 'none'
    warrior = 'warrior'
    spellcaster = 'spellcaster'
    fairy = 'fairy'
    fiend = 'fiend'
    zombie = 'zombie'
    machine = 'machine'
    aqua = 'aqua'
    pyro = 'pyro'
    rock = 'rock'
    windbeast = 'windbeast'
    plant = 'plant'
    insect = 'insect'
    thunder = 'thunder'
    dragon = 'dragon'
    beast = 'beast'
    beast_warrior = 'beast_warrior'
    dinosaur = 'dinosaur'
    fish = 'fish'
    sea_serpent = 'sea_serpent'
    reptile = 'reptile'
    psycho = 'psycho'
    devine = 'devine'
    creator_god = 'creator_god'
    wyrm = 'wyrm'
    cyberse = 'cyberse'
    illusion = 'illusion'


class Type(Enum):
    monster = 'monster'
    spell = 'spell'
    trap = 'trap'
    normal = 'normal'
    effect = 'effect'
    fusion = 'fusion'
    ritual = 'ritual'
    trap_monster = 'trap_monster'
    spirit = 'spirit'
    union = 'union'
    dual = 'dual'
    tuner = 'tuner'
    synchro = 'synchro'
    token = 'token'
    quick_play = 'quick_play'
    continuous = 'continuous'
    equip = 'equip'
    field = 'field'
    counter = 'counter'
    flip = 'flip'
    toon = 'toon'
    xyz = 'xyz'
    pendulum = 'pendulum'
    special = 'special'
    link = 'link'


class Card(BaseModel):
    code: int = Field(
        ..., description='Card code from cards.cdb', examples=[23434538, 23995346]
    )
    location: Location = Field(..., examples=['hand', 'deck'])
    sequence: int = Field(
        ...,
        description='Sequence in ocgcore, 0 is N/A or unknown, if not, shoud start from 1. Only non-zero for cards in mzone, szone and grave.',
        examples=[1],
    )
    controller: Controller = Field(..., examples=['me'])
    position: Position = Field(
        ...,
        description='If the monster is xyz material (overlay_sequence != -1), the position is faceup.',
        examples=['faceup_attack'],
    )
    overlay_sequence: int = Field(..., description='if is overlay, this is the overlay index, starting from 0, else -1.')
    attribute: Attribute = Field(
        ..., description='none for N/A or unknown or token.', examples=['earth']
    )
    race: Race = Field(
        ..., description='none for N/A or unknown or token.', examples=['fish']
    )
    level: int = Field(
        ...,
        description='Rank and link are also considered as level. 0 is N/A or unknown.',
        examples=[4],
    )
    counter: int = Field(
        ...,
        description='Number of counters. If there are 2 types of counters or more, we consider only the first type of counter.',
        examples=[1],
    )
    negated: bool = Field(
        ..., description='Whether the card effect is disabled or forbidden'
    )
    attack: int = Field(..., examples=[3000])
    defense: int = Field(..., examples=[2500])
    types: List[Type] = Field(..., min_items=0)


class Phase(Enum):
    draw = 'draw'
    standby = 'standby'
    main1 = 'main1'
    battle_start = 'battle_start'
    battle_step = 'battle_step'
    damage = 'damage'
    damage_calculation = 'damage_calculation'
    battle = 'battle'
    main2 = 'main2'
    end = 'end'


class Global(BaseModel):
    my_lp: int = Field(..., examples=[8000])
    op_lp: int = Field(..., examples=[8000])
    turn: int = Field(..., examples=[1])
    phase: Phase
    is_first: bool = Field(..., description='Whether me is the first player')
    is_my_turn: bool


class MsgName(Enum):
    select_idlecmd = 'select_idlecmd'
    select_chain = 'select_chain'
    select_card = 'select_card'
    select_tribute = 'select_tribute'
    select_position = 'select_position'
    select_effectyn = 'select_effectyn'
    select_yesno = 'select_yesno'
    select_battlecmd = 'select_battlecmd'
    select_unselect_card = 'select_unselect_card'
    select_option = 'select_option'
    select_place = 'select_place'
    select_sum = 'select_sum'
    select_disfield = 'select_disfield'
    announce_attrib = 'announce_attrib'
    announce_number = 'announce_number'


class SelectUnselectCard(BaseModel):
    location: CardLocation
    response: int


class MsgSelectUnselectCard(BaseModel):
    msg_type: Literal['select_unselect_card']
    finishable: bool
    cancelable: bool = Field(..., description='ignored')
    min: int
    max: int
    selected_cards: List[SelectUnselectCard] = Field(..., description='ignored')
    selectable_cards: List[SelectUnselectCard]


class BattleCmdType(Enum):
    attack = 'attack'
    activate = 'activate'
    to_m2 = 'to_m2'
    to_ep = 'to_ep'


class BattleCmdData(BaseModel):
    card_info: CardInfo
    effect_description: int
    direct_attackable: bool
    response: int


class BattleCmd(BaseModel):
    cmd_type: BattleCmdType
    data: Optional[BattleCmdData] = None


class MsgSelectBattleCmd(BaseModel):
    msg_type: Literal['select_battlecmd']
    battle_cmds: List[BattleCmd]


class ActionMsg(BaseModel):
    data: Union[
        MsgSelectCard,
        MsgSelectTribute,
        MsgSelectSum,
        MsgSelectIdleCmd,
        MsgSelectChain,
        MsgSelectPosition,
        MsgSelectEffectYn,
        MsgSelectYesNo,
        MsgSelectBattleCmd,
        MsgSelectUnselectCard,
        MsgSelectOption,
        MsgSelectPlace,
        MsgSelectDisfield,
        MsgAnnounceAttrib,
        MsgAnnounceNumber,
    ] = Field(..., discriminator='msg_type')


class Input(BaseModel):
    global_: Global = Field(..., alias='global')
    cards: List[Card] = Field(..., max_items=160, min_items=2)
    action_msg: ActionMsg


class DuelPredictRequest(BaseModel):
    input: Input
    prev_action_idx: int = Field(
        ...,
        description='The index of the previous action. It should be 0 for the first action.',
    )
    index: int = Field(
        ...,
        description='The index must be equal to the index from the previous response of the same duelId.',
    )


class DuelPredictResponse(BaseModel):
    predict_results: MsgResponse
    index: int = Field(
        ..., description="It will be equal to the request's index + 1.", examples=[1]
    )


class DuelPredictErrorResponse(BaseModel):
    error: str = Field(..., description='error message', examples=['index conflict'])


class DuelCreateResponse(BaseModel):
    duelId: str = Field(..., description='The duel id', examples=['007f8d84-7944-4851-921c-d61d4884a841'])
    index: int = Field(..., description='The index must be 0.')
