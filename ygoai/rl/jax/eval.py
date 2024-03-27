import numpy as np


def evaluate(envs, act_fn, params):
    num_episodes = envs.num_envs
    episode_lengths = []
    episode_rewards = []
    eval_win_rates = []
    obs = envs.reset()[0]
    while True:
        actions = act_fn(params, obs)
        actions = np.array(actions)

        obs, rewards, dones, info = envs.step(actions)

        for idx, d in enumerate(dones):
            if not d:
                continue
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