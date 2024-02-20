# Features

## Card (39)
- id: 2, uint16 -> 2 uint8, name+desc
- location: 1, int, 0: N/A, 1+: same as location2str (9)
- seq: 1, int, 0: N/A, 1+: seq in location
- owner: 1, int, 0: me, 1: oppo (2)
- position: 1, int, 0: N/A, same as position2str
- overlay: 1, int, 0: not, 1: xyz material
- attribute: 1, int, 0: N/A, same as attribute2str[2:]
- race: 1, int, 0: N/A, same as race2str
- level: 1, int, 0: N/A
- atk: 2, max 65535 to 2 bytes
- def: 2, max 65535 to 2 bytes
- type: 25, multi-hot, same as type2str

## Global
- lp: 2, max 65535 to 2 bytes
- oppo_lp: 2, max 65535 to 2 bytes
- turn: 1, int, trunc to 8
- phase: 1, int, one-hot (10)
- is_first: 1, int, 0: False, 1: True
- is_my_turn: 1, int, 0: False, 1: True
- is_end: 1, int, 0: False, 1: True


## Legal Actions (max 8)
- spec index: 8, int, select target
- msg: 1, int (16)
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
- cancel_finish: 1, int (3)
  - N/A
  - Cancel
  - Finish
- position: 1, int , 0: N/A, same as position2str
- option: 1, int, 0: N/A
- place: 1, int (31), 0: N/A,
  - 1-7: m
  - 8-15: s
  - 16-22: om
  - 23-30: os
- attribute: 1, int, 0: N/A, same as attribute2id


## History Actions
- id: 2x4, uint16 -> 2 uint8, name+desc
- same as Legal Actions
