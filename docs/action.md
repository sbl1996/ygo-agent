# Action

## Types
- Set + card
- Reposition + card
- Special summon + card
- Summon Face-up Attack + card
- Summon Face-down Defense + card
- Attack + card
- DirectAttack + card
- Activate + card + effect
- Cancel
- Switch + phase
- SelectPosition + card + position
- AnnounceNumber + card + effect + number
- SelectPlace + card + place
- AnnounceAttrib + card + effect + attrib

## Effect

### MSG_SELECT_BATTLECMD | MSG_SELECT_IDLECMD | MSG_SELECT_CHAIN | MSG_SELECT_EFFECTYN
- desc == 0: default effect of card
- desc < LIMIT: system string
- desc > LIMIT: card + effect

### MSG_SELECT_OPTION | MSG_SELECT_YESNO
- desc == 0: error
- desc < LIMIT: system string
- desc > LIMIT: card + effect
