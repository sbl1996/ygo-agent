from typing import List, Union
from dataclasses import dataclass

import sqlite3

import pandas as pd

from ygoai.constants import TYPE, type2str, attribute2str, race2str


def parse_types(value):
    types = []
    all_types = list(type2str.keys())
    for t in all_types:
        if value & t:
            types.append(type2str[t])
    return types


def parse_attribute(value):
    attribute = attribute2str.get(value, None)
    assert attribute, "Invalid attribute, value: " + str(value)
    return attribute


def parse_race(value):
    race = race2str.get(value, None)
    assert race, "Invalid race, value: " + str(value)
    return race


@dataclass
class Card:
    code: int
    name: str
    desc: str
    types: List[str]

    def format(self):
        return format_card(self)


@dataclass
class MonsterCard(Card):
    atk: int
    def_: int
    level: int
    race: str
    attribute: str


@dataclass
class SpellCard(Card):
    pass


@dataclass
class TrapCard(Card):
    pass


def format_monster_card(card: MonsterCard):
    name = card.name
    typ = "/".join(card.types)

    attribute = card.attribute
    race = card.race

    level = str(card.level)

    atk = str(card.atk)
    if atk == '-2':
        atk = '?'

    def_ = str(card.def_)
    if def_ == '-2':
        def_ = '?'

    if typ == 'Monster/Normal':
        desc = "-"
    else:
        desc = card.desc

    columns = [name, typ, attribute, race, level, atk, def_, desc]
    return " | ".join(columns)


def format_spell_trap_card(card: Union[SpellCard, TrapCard]):
    name = card.name
    typ = "/".join(card.types)
    desc = card.desc

    columns = [name, typ, desc]
    return " | ".join(columns)


def format_card(card: Card):
    if isinstance(card, MonsterCard):
        return format_monster_card(card)
    elif isinstance(card, (SpellCard, TrapCard)):
        return format_spell_trap_card(card)
    else:
        raise ValueError("Invalid card type: " + str(card))


## For analyzing cards.db

def parse_monster_card(data) -> MonsterCard:
    code = int(data['id'])
    name = data['name']
    desc = data['desc']
    
    types = parse_types(int(data['type']))
    
    atk = int(data['atk'])
    def_ = int(data['def'])
    level = int(data['level'])

    if level >= 16:
        # pendulum monster
        level = level % 16

    race = parse_race(int(data['race']))
    attribute = parse_attribute(int(data['attribute']))
    return MonsterCard(code, name, desc, types, atk, def_, level, race, attribute)


def parse_spell_card(data) -> SpellCard:
    code = int(data['id'])
    name = data['name']
    desc = data['desc']
    
    types = parse_types(int(data['type']))
    return SpellCard(code, name, desc, types)


def parse_trap_card(data) -> TrapCard:
    code = int(data['id'])
    name = data['name']
    desc = data['desc']
    
    types = parse_types(int(data['type']))
    return TrapCard(code, name, desc, types)


def parse_card(data) -> Card:
    type_ = data['type']
    if type_ & TYPE.MONSTER:
        return parse_monster_card(data)
    elif type_ & TYPE.SPELL:
        return parse_spell_card(data)
    elif type_ & TYPE.TRAP:
        return parse_trap_card(data)
    else:
        raise ValueError("Invalid card type: " + str(type_))


def read_cards(cards_path):
    conn = sqlite3.connect(cards_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM datas")
    datas_rows = cursor.fetchall()
    datas_columns = [description[0] for description in cursor.description]
    datas_df = pd.DataFrame(datas_rows, columns=datas_columns)

    cursor.execute("SELECT * FROM texts")
    texts_rows = cursor.fetchall()
    texts_columns = [description[0] for description in cursor.description]
    texts_df = pd.DataFrame(texts_rows, columns=texts_columns)

    cursor.close()
    conn.close()

    texts_df = texts_df.loc[:, ['id', 'name', 'desc']]
    merged_df = pd.merge(texts_df, datas_df, on='id')

    cards_data = merged_df.to_dict('records')
    cards = [parse_card(data) for data in cards_data]
    return merged_df, cards
