# Features

## Definitions
- float transform: max 65535 -> 2 bytes
- count

## Card (39)
- 0,1: card id, uint16 -> 2 uint8, name+desc
- 2: location, discrete, 0: N/A, 1+: same as location2str (9)
- 3: seq, discrete, 0: N/A, 1+: seq in location
- 4: owner, discrete, 0: me, 1: oppo (2)
- 5: position, discrete, 0: N/A, 1+: same as position2str
- 6: overlay, discrete, 0: not, 1: xyz material
- 7: attribute, discrete, 0: N/A, 1+: same as attribute2str[2:]
- 8: race, discrete, 0: N/A, 1+: same as race2str
- 9: level, discrete, 0: N/A
- 10: counter, discrete, 0: N/A
- 11: negated, discrete, 0: False, 1: True
- 12,13: atk, float transform
- 14,15: def: float transform
- 16-40: type, multi-hot, same as type2str (25)

## Global
- 0,1: my_lp, float transform
- 2,3: op_lp, float transform
- 4: turn, discrete, trunc to 16
- 5: phase, discrete (10)
- 6: is_first, discrete, 0: False, 1: True
- 7: is_my_turn, discrete, 0: False, 1: True
- 8: n_my_decks, count
- 9: n_my_hands, count
- 10: n_my_monsters, count
- 11: n_my_spell_traps, count
- 12: n_my_graves, count
- 13: n_my_removes, count
- 14: n_my_extras, count
- 15: n_op_decks, count
- 16: n_op_hands, count
- 17: n_op_monsters, count
- 18: n_op_spell_traps, count
- 19: n_op_graves, count
- 20: n_op_removes, count
- 21: n_op_extras, count
- 22: is_end, discrete, 0: False, 1: True


## Legal Actions (max 24)
- 0,1: spec index or card id, uint16 -> 2 uint8
- 2: msg, discrete, 0: N/A, 1+: same as msg2str (11)
- act: 1, int (11)
  - N/A
  - t: Set
  - r: Reposition
  - c: Special Summon
  - s: Summon Face-up Attack
  - m: Summon Face-down Defense
  - a: Attack
  - v: Activate
  - v2: Activate the second effect
  - v3: Activate the third effect
  - v4: Activate the fourth effect
- yes/no: 1, int (3)
  - N/A
  - Yes
  - No
- phase: 1, int (4)
  - N/A
  - Battle (b)
  - Main Phase 2 (m)
  - End Phase (e)
- cancel: 1
  - N/A
  - Cancel
- finish: 1
  - N/A
  - Finish
- position: 1, int , 0: N/A, same as position2str
- option: 1, int, 0: N/A
- number: 1, int, 0: N/A
- place: 1, int (31), 0: N/A,
  - 1-7: m
  - 8-15: s
  - 16-22: om
  - 23-30: os
- attribute: 1, int, 0: N/A, same as attribute2id


## History Actions
- id: 2x4, uint16 -> 2 uint8, name+desc
- same as Legal Actions
