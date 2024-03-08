import numpy as np
import torch
from torch.cuda.amp import autocast

from ygoai.rl.utils import to_tensor


def evaluate(envs, model, num_episodes, device, fp16_eval=False):
    episode_lengths = []
    episode_rewards = []
    eval_win_rates = []
    obs = envs.reset()[0]
    while True:
        obs = to_tensor(obs, device, dtype=torch.uint8)
        with torch.no_grad():
            with autocast(enabled=fp16_eval):
                logits = model(obs)[0]
        probs = torch.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()
        actions = probs.argmax(axis=1)

        obs, rewards, dones, info = envs.step(actions)

        for idx, d in enumerate(dones):
            if d:
                episode_length = info['l'][idx]
                episode_reward = info['r'][idx]
                win = 1 if episode_reward > 0 else 0

                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                eval_win_rates.append(win)
        if len(episode_lengths) >= num_episodes:
            break
    
    eval_return = np.mean(episode_rewards[:num_episodes])
    eval_ep_len = np.mean(episode_lengths[:num_episodes])
    eval_win_rate = np.mean(eval_win_rates[:num_episodes])
    return eval_return, eval_ep_len, eval_win_rate