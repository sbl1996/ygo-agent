# Change log

## 0.2.0 (March 12, 2024)
- A feature of negated is added to cards. This feature is used to indicate whether a card is negated or not.
- Positional encoding is added to history actions. When it wasn't added before, the model cannot distinguish the order of history actions.
- Multi-selet action is removed and implemented by multiple single-select actions. It means that the number of selections is now unlimited.
