import numpy as np


def evaluate(envs, num_episodes, predict_fn, rnn_state=None):
    episode_lengths = []
    episode_rewards = []
    win_rates = []
    obs = envs.reset()[0]
    collected = np.zeros((num_episodes,), dtype=np.bool_)
    while True:
        if rnn_state is None:
            actions = predict_fn(obs)
        else:
            rnn_state, actions = predict_fn(obs, rnn_state)
        actions = np.array(actions)

        obs, rewards, dones, info = envs.step(actions)

        for idx, d in enumerate(dones):
            if not d or collected[idx]:
                continue
            collected[idx] = True
            episode_length = info['l'][idx]
            episode_reward = info['r'][idx]
            win = 1 if episode_reward > 0 else 0

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            win_rates.append(win)
        if len(episode_lengths) >= num_episodes:
            break
    
    eval_return = np.mean(episode_rewards[:num_episodes])
    eval_ep_len = np.mean(episode_lengths[:num_episodes])
    eval_win_rate = np.mean(win_rates[:num_episodes])
    return eval_return, eval_ep_len, eval_win_rate


def battle(envs, num_episodes, predict_fn, rstate1=None, rstate2=None):
    assert num_episodes == envs.num_envs
    num_envs = envs.num_envs
    episode_rewards = []
    episode_lengths = []
    win_rates = []

    obs, infos = envs.reset()
    next_to_play = infos['to_play']
    dones = np.zeros(num_envs, dtype=np.bool_)
    collected = np.zeros((num_episodes,), dtype=np.bool_)

    main_player = np.concatenate([
        np.zeros(num_envs // 2, dtype=np.int64),
        np.ones(num_envs - num_envs // 2, dtype=np.int64)
    ])

    while True:
        main = next_to_play == main_player
        rstate1, rstate2, actions = predict_fn(obs, rstate1, rstate2, main, dones)
        actions = np.array(actions)

        obs, rewards, dones, infos = envs.step(actions)
        next_to_play = infos['to_play']

        for idx, d in enumerate(dones):
            if not d or collected[idx]:
                continue
            collected[idx] = True
            episode_length = infos['l'][idx]
            episode_reward = infos['r'][idx] * (1 if main[idx] else -1)
            win = 1 if episode_reward > 0 else 0

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            win_rates.append(win)
        if len(episode_lengths) >= num_episodes:
            break

    eval_return = np.mean(episode_rewards[:num_episodes])
    eval_ep_len = np.mean(episode_lengths[:num_episodes])
    eval_win_rate = np.mean(win_rates[:num_episodes])
    return eval_return, eval_ep_len, eval_win_rate
