# Deck

## Unsupported
- Many (Crossout Designator)
- Magician (pendulum)
- Shiranui (Fairy Tail - Snow)

# Messgae

## add_counter
Not supported

## select_card
- `min` and `max` <= 5 are supported
- `min` > 5 throws an error
- `max` > 5 is truncated to 5

### related cards
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

## announce_number
Only 1-12 is supported.
