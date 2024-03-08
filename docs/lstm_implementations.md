# LSTM Implementations

## Original PPO + LSTM in CleanRL
```python
not_done = (~done.reshape((-1, batch_size))).float()
new_hidden = []
for i in range(hidden.shape[0]):
    h, lstm_state = self.lstm(
        hidden[i].unsqueeze(0),
        (
            not_done[i].view(1, -1, 1) * lstm_state[0],
            not_done[i].view(1, -1, 1) * lstm_state[1],
        ),
    )
    new_hidden += [h]
new_hidden = torch.cat(new_hidden)

# new_hidden, lstm_state = self.lstm(hidden, lstm_state)
```
The length of the loop is the `num_steps` (typically 128), therefore it is slow (even with torch.compile). Compared with the original LSTM, the overall training time is 4x slower.

## Custom LSTM with triton
```python
```