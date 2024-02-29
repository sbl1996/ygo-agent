# Dimensions
B: batch size
C: number of channels
H: number of history actions

# Features
f_cards: (B, n_cards, C), features of cards
f_global: (B, C), global features
f_h_actions: (B, H, C), features of history actions
f_actions: (B, max_actions, C), features of current legal actions

output: (B, max_actions, 1), value of each action

# Fusion

## Method 1
```
f_cards -> n encoder layers -> f_cards
f_global -> ResMLP -> f_global
f_cards = f_cards + f_global
f_actions -> n encoder layers -> f_actions

f_cards[id] -> f_a_cards -> ResMLP -> f_a_cards
f_actions = f_a_cards + f_a_feats

f_actions, f_cards -> n decoder layers -> f_actions

f_h_actions -> n encoder layers -> f_h_actions
f_actions, f_h_actions -> n decoder layers -> f_actions

f_actions -> MLP -> output
```