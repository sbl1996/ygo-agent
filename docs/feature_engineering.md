# Features

## Definitions

### Float transform
- float transform: max 65535 -> 2 bytes

### Card ID
The card id is the index of the card code in `code_list.txt`.

## Card
- 0,1: card id, uint16 -> 2 uint8, name+desc
- 2: location, discrete, 0: N/A, 1+: same as location2str (9)
- 3: seq, discrete, 0: N/A, 1+: seq in location
- 4: owner, discrete, 0: me, 1: oppo (2)
- 5: position, discrete, 0: N/A, 1+: same as position2str
- 6: overlay, discrete, 0: not, 1: xyz material
- 7: attribute, discrete, 0: N/A, 1+: same as attribute2str
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


## Legal Actions
- 0: spec index
- 1,2: code, uint16 -> 2 uint8
- 3: msg, discrete, 0: N/A, 1+: same as msg2str (15)
- 4: act, discrete (11)
  - N/A
  - Set
  - Reposition
  - Special Summon
  - Summon Face-up Attack
  - Summon Face-down Defense
  - Attack
  - DirectAttack
  - Activate
  - Cancel
- 5: finish, discrete (2)
  - N/A
  - Finish
- 6: effect, discrete, 0: N/A
- 7: phase, discrete (4)
  - N/A
  - Battle (b)
  - Main Phase 2 (m)
  - End Phase (e)
- 8: position, discrete, 0: N/A, same as position2str
- 9: number, discrete, 0: N/A
- 10: place, discrete
  - 0: N/A
  - 1-7: m
  - 8-15: s
  - 16-22: om
  - 23-30: os
- 11: attribute, discrete, 0: N/A, same as attribute2id


## History Actions
- 0,1: card id, uint16 -> 2 uint8
- 2-11 same as legal actions
- 12: turn, discrete, trunc to 3
- 13: phase, discrete (10)
