from enum import Flag, auto, unique, IntFlag

__ = lambda s: s

AMOUNT_ATTRIBUTES = 7
AMOUNT_RACES = 25

ATTRIBUTES_OFFSET = 1010


LINK_MARKERS = {
	0x001: __("bottom left"),
	0x002: __("bottom"),
	0x004: __("bottom right"),
	0x008: __("left"),
	0x020: __("right"),
	0x040: __("top left"),
	0x080: __("top"),
	0x100: __("top right")
}

@unique
class LOCATION(IntFlag):
	DECK = 0x1
	HAND = 0x2
	MZONE = 0x4
	SZONE = 0x8
	GRAVE = 0x10
	REMOVED = 0x20
	EXTRA = 0x40
	OVERLAY = 0x80
	ONFIELD = MZONE | SZONE
	FZONE = 0x100
	PZONE = 0x200
	DECKBOT = 0x10001		# Return to deck bottom
	DECKSHF	= 0x20001		# Return to deck and shuffle

location2str = {
    LOCATION.DECK: 'Deck',
    LOCATION.HAND: 'Hand',
    LOCATION.MZONE: 'Main Monster Zone',
    LOCATION.SZONE: 'Spell & Trap Zone',
    LOCATION.GRAVE: 'Graveyard',
    LOCATION.REMOVED: 'Banished',
    LOCATION.EXTRA: 'Extra Deck',
    LOCATION.FZONE: 'Field Zone',
}

all_locations = list(location2str.keys())


PHASES = {
	0x01: __('draw phase'),
	0x02: __('standby phase'),
	0x04: __('main1 phase'),
	0x08: __('battle start phase'),
	0x10: __('battle step phase'),
	0x20: __('damage phase'),
	0x40: __('damage calculation phase'),
	0x80: __('battle phase'),
	0x100: __('main2 phase'),
	0x200: __('end phase'),
}

@unique
class POSITION(IntFlag):
	FACEUP_ATTACK = 0x1
	FACEDOWN_ATTACK = 0x2
	FACEUP_DEFENSE = 0x4
	FACEUP = FACEUP_ATTACK | FACEUP_DEFENSE
	FACEDOWN_DEFENSE = 0x8
	FACEDOWN = FACEDOWN_ATTACK | FACEDOWN_DEFENSE
	ATTACK = FACEUP_ATTACK | FACEDOWN_ATTACK
	DEFENSE = FACEUP_DEFENSE | FACEDOWN_DEFENSE


position2str = {
	POSITION.FACEUP_ATTACK: "Face-up Attack",
	POSITION.FACEDOWN_ATTACK: "Face-down Attack",
	POSITION.FACEUP_DEFENSE: "Face-up Defense",
	POSITION.FACEUP: "Face-up",
	POSITION.FACEDOWN_DEFENSE: "Face-down Defense",
	POSITION.FACEDOWN: "Face-down",
	POSITION.ATTACK: "Attack",
	POSITION.DEFENSE: "Defense",
}

all_positions = list(position2str.keys())


RACES_OFFSET = 1020

@unique
class QUERY(IntFlag):
	CODE = 0x1
	POSITION = 0x2
	ALIAS = 0x4
	TYPE = 0x8
	LEVEL = 0x10
	RANK = 0x20
	ATTRIBUTE = 0x40
	RACE = 0x80
	ATTACK = 0x100
	DEFENSE = 0x200
	BASE_ATTACK = 0x400
	BASE_DEFENSE = 0x800
	REASON = 0x1000
	REASON_CARD = 0x2000
	EQUIP_CARD = 0x4000
	TARGET_CARD = 0x8000
	OVERLAY_CARD = 0x10000
	COUNTERS = 0x20000
	OWNER = 0x40000
	STATUS = 0x80000
	LSCALE = 0x200000
	RSCALE = 0x400000
	LINK = 0x800000

@unique
class TYPE(IntFlag):
	MONSTER = 0x1
	SPELL = 0x2
	TRAP = 0x4
	NORMAL = 0x10
	EFFECT = 0x20
	FUSION = 0x40
	RITUAL = 0x80
	TRAPMONSTER = 0x100
	SPIRIT = 0x200
	UNION = 0x400
	DUAL = 0x800
	TUNER = 0x1000
	SYNCHRO = 0x2000
	TOKEN = 0x4000
	QUICKPLAY = 0x10000
	CONTINUOUS = 0x20000
	EQUIP = 0x40000
	FIELD = 0x80000
	COUNTER = 0x100000
	FLIP = 0x200000
	TOON = 0x400000
	XYZ = 0x800000
	PENDULUM = 0x1000000
	SPSUMMON = 0x2000000
	LINK = 0x4000000
	# for this mud only
	EXTRA = XYZ | SYNCHRO | FUSION | LINK


type2str = {
    TYPE.MONSTER: "Monster",
    TYPE.SPELL: "Spell",
    TYPE.TRAP: "Trap",
    TYPE.NORMAL: "Normal",
    TYPE.EFFECT: "Effect",
    TYPE.FUSION: "Fusion",
    TYPE.RITUAL: "Ritual",
    TYPE.TRAPMONSTER: "Trap Monster",
    TYPE.SPIRIT: "Spirit",
    TYPE.UNION: "Union",
    TYPE.DUAL: "Dual",
    TYPE.TUNER: "Tuner",
    TYPE.SYNCHRO: "Synchro",
    TYPE.TOKEN: "Token",
    TYPE.QUICKPLAY: "Quick-play",
    TYPE.CONTINUOUS: "Continuous",
    TYPE.EQUIP: "Equip",
    TYPE.FIELD: "Field",
    TYPE.COUNTER: "Counter",
    TYPE.FLIP: "Flip",
    TYPE.TOON: "Toon",
    TYPE.XYZ: "XYZ",
    TYPE.PENDULUM: "Pendulum",
    TYPE.SPSUMMON: "Special",
    TYPE.LINK: "Link"
}

all_types = list(type2str.keys())


@unique
class ATTRIBUTE(IntFlag):
	ALL = 0x7f
	NONE = 0x0  # Token
	EARTH = 0x01
	WATER = 0x02
	FIRE = 0x04
	WIND = 0x08
	LIGHT = 0x10
	DARK = 0x20
	DEVINE = 0x40	


attribute2str = {
    ATTRIBUTE.ALL: 'All',
    ATTRIBUTE.NONE: 'None',
    ATTRIBUTE.EARTH: 'Earth',
    ATTRIBUTE.WATER: 'Water',
    ATTRIBUTE.FIRE: 'Fire',
    ATTRIBUTE.WIND: 'Wind',
    ATTRIBUTE.LIGHT: 'Light',
    ATTRIBUTE.DARK: 'Dark',
    ATTRIBUTE.DEVINE: 'Divine'
}

all_attributes = list(attribute2str.keys())


@unique
class RACE(IntFlag):
	ALL = 0x3ffffff
	NONE = 0x0  # Token
	WARRIOR = 0x1
	SPELLCASTER = 0x2
	FAIRY = 0x4
	FIEND = 0x8
	ZOMBIE = 0x10
	MACHINE = 0x20
	AQUA = 0x40
	PYRO = 0x80
	ROCK = 0x100
	WINDBEAST = 0x200
	PLANT = 0x400
	INSECT = 0x800
	THUNDER = 0x1000
	DRAGON = 0x2000
	BEAST = 0x4000
	BEASTWARRIOR = 0x8000
	DINOSAUR = 0x10000
	FISH = 0x20000
	SEASERPENT = 0x40000
	REPTILE = 0x80000
	PSYCHO = 0x100000
	DEVINE = 0x200000
	CREATORGOD = 0x400000
	WYRM = 0x800000
	CYBERSE = 0x1000000
	ILLUSION = 0x2000000


race2str = {
    RACE.NONE: "None",
    RACE.WARRIOR: 'Warrior',
    RACE.SPELLCASTER: 'Spellcaster',
    RACE.FAIRY: 'Fairy',
    RACE.FIEND: 'Fiend',
    RACE.ZOMBIE: 'Zombie',
    RACE.MACHINE: 'Machine',
    RACE.AQUA: 'Aqua',
    RACE.PYRO: 'Pyro',
    RACE.ROCK: 'Rock',
    RACE.WINDBEAST: 'Windbeast',
    RACE.PLANT: 'Plant',
    RACE.INSECT: 'Insect',
    RACE.THUNDER: 'Thunder',
    RACE.DRAGON: 'Dragon',
    RACE.BEAST: 'Beast',
    RACE.BEASTWARRIOR: 'Beast Warrior',
    RACE.DINOSAUR: 'Dinosaur',
    RACE.FISH: 'Fish',
    RACE.SEASERPENT: 'Sea Serpent',
    RACE.REPTILE: 'Reptile',
    RACE.PSYCHO: 'Psycho',
    RACE.DEVINE: 'Divine',
    RACE.CREATORGOD: 'Creator God',
    RACE.WYRM: 'Wyrm',
    RACE.CYBERSE: 'Cyberse',
    RACE.ILLUSION: 'Illusion'
}

all_races = list(race2str.keys())


@unique
class REASON(IntFlag):
	DESTROY = 0x1
	RELEASE = 0x2
	TEMPORARY = 0x4
	MATERIAL = 0x8
	SUMMON = 0x10
	BATTLE = 0x20
	EFFECT = 0x40
	COST = 0x80
	ADJUST = 0x100
	LOST_TARGET = 0x200
	RULE = 0x400
	SPSUMMON = 0x800
	DISSUMMON = 0x1000
	FLIP = 0x2000
	DISCARD = 0x4000
	RDAMAGE = 0x8000
	RRECOVER = 0x10000
	RETURN = 0x20000
	FUSION = 0x40000
	SYNCHRO = 0x80000
	RITUAL = 0x100000
	XYZ = 0x200000
	REPLACE = 0x1000000
	DRAW = 0x2000000
	REDIRECT = 0x4000000
	REVEAL = 0x8000000
	LINK = 0x10000000
	LOST_OVERLAY = 0x20000000
	MAINTENANCE = 0x40000000
	ACTION = 0x80000000
	PROCEDURE = SYNCHRO | XYZ | LINK

@unique
class OPCODE(IntFlag):
	ADD = 0x40000000
	SUB = 0x40000001
	MUL = 0x40000002
	DIV = 0x40000003
	AND = 0x40000004
	OR = 0x40000005
	NEG = 0x40000006
	NOT = 0x40000007
	ISCODE = 0x40000100
	ISSETCARD = 0x40000101
	ISTYPE = 0x40000102
	ISRACE = 0x40000103
	ISATTRIBUTE = 0x40000104


@unique
class INFORM(Flag):
	PLAYER = auto()
	OPPONENT = auto()
	ALL = PLAYER | OPPONENT

@unique
class DECK(Flag):
	OWNED = auto()
	OTHER = auto()
	PUBLIC = auto()
	ALL = OWNED | OTHER  # should only be used for admins
	VISIBLE = OWNED | PUBLIC  # default scope for players