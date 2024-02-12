# Deck

## Unsupported
- Many (Crossout Designator)
- Blackwing (add_counter)
- Magician (pendulum)
- Shaddoll (add_counter)
- Shiranui (Fairy Tail - Snow)
- Hero (random_selected)

# Messgae

## random_selected
Not supported

## add_counter
Not supported

## select_card
- `min` and `max` <= 5 are supported
- `min` > 5 throws an error
- `max` > 5 is truncated to 5

### Unsupported
- Fairy Tail - Snow (min=max=7)
- Pot of Prosperity (min=max=6)

## announce_card
Not supported:
- Alsei, the Sylvan High Protector
- Crossout Designator

## announce_attrib
Only 1 attribute is announced at a time.
Not supported:
- DNA Checkup

# Summon

## Tribute Summon
Through `select_tribute` (multi-select)

## Link Summon
Through `select_unselect_card` (select 1 card per time)

## Syncro Summon
- `select_card` to choose the tuner (usually 1 card)
- `select_sum` to choose the non-tuner (1 card per time)
