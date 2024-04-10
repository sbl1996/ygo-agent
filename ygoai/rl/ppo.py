import numpy as np

import torch
from torch.cuda.amp import autocast
import torch_xla.core.xla_model as xm

from ygoai.rl.utils import masked_normalize, masked_mean


def entropy_from_logits(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * torch.softmax(logits, dim=-1)
    return -p_log_p.sum(-1)


def train_step(agent, optimizer, scaler, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_learns, args):
    with autocast(enabled=args.fp16_train):
        logits, newvalue, valid = agent(mb_obs)[:3]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        newlogprob = logits.gather(-1, mb_actions[:, None]).squeeze(-1)
        entropy = entropy_from_logits(logits)
    valid = torch.logical_and(valid, mb_learns)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        mb_advantages = masked_normalize(mb_advantages, valid, eps=1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2)
    pg_loss = masked_mean(pg_loss, valid)

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
    v_loss = masked_mean(v_loss, valid)

    entropy_loss = masked_mean(entropy, valid)
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    optimizer.zero_grad()
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    return old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss


# def train_step_t(agent, optimizer, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_learns, args):
def train_step_t(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, b_learns, mb_inds, args):
    mb_obs = {
        k: v[mb_inds] for k, v in b_obs.items()
    }
    mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_learns = [
        v[mb_inds] for v in [b_actions, b_logprobs, b_advantages, b_returns, b_values, b_learns]]

    optimizer.zero_grad(True)
    logits, newvalue, valid = agent(mb_obs)
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    newlogprob = logits.gather(-1, mb_actions[:, None]).squeeze(-1)
    entropy = entropy_from_logits(logits)
    valid = torch.logical_and(valid, mb_learns)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        mb_advantages = masked_normalize(mb_advantages, valid, eps=1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2)
    pg_loss = masked_mean(pg_loss, valid)

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
    v_loss = masked_mean(v_loss, valid)

    entropy_loss = masked_mean(entropy, valid)
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    loss.backward()
    xm.optimizer_step(optimizer)
    return old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss


def bootstrap_value(values, rewards, dones, nextvalues, next_done, gamma, gae_lambda):
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = nextvalues
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam


def bootstrap_value_self(values, rewards, dones, learns, nextvalues, next_done, gamma, gae_lambda):
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    done_used = torch.ones_like(next_done, dtype=torch.bool)
    reward = 0
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        # if learns[t]:
        #     if dones[t+1]:
        #         reward = rewards[t]
        #         nextvalues = 0
        #         lastgaelam = 0
        #         done_used = True
        #     else:
        #         if not done_used:
        #             reward = reward
        #             nextvalues = 0
        #             lastgaelam = 0
        #             done_used = True
        #         else:
        #             reward = rewards[t]
        #     delta = reward + args.gamma * nextvalues - values[t]
        #     lastgaelam_ = delta + args.gamma * args.gae_lambda * lastgaelam
        #     advantages[t] = lastgaelam_
        #     nextvalues = values[t]
        #     lastgaelam = lastgaelam_
        # else:
        #     if dones[t+1]:
        #         reward = -rewards[t]
        #         done_used = False
        #     else:
        #         reward = reward
        learn = learns[t]
        if t != num_steps - 1:
            next_done = dones[t + 1]
        sp = 2 * (learn.int() - 0.5)
        reward = torch.where(next_done, rewards[t] * sp, torch.where(learn & done_used, 0, reward))
        real_done = next_done | ~done_used
        nextvalues = torch.where(real_done, 0, nextvalues)
        lastgaelam = torch.where(real_done, 0, lastgaelam)
        done_used = torch.where(
            next_done, learn, torch.where(learn & ~done_used, True, done_used))

        delta = reward + gamma * nextvalues - values[t]
        lastgaelam_ = delta + gamma * gae_lambda * lastgaelam
        advantages[t] = lastgaelam_
        nextvalues = torch.where(learn, values[t], nextvalues)
        lastgaelam = torch.where(learn, lastgaelam_, lastgaelam)
    return advantages


def bootstrap_value_selfplay(values, rewards, dones, learns, nextvalues1, nextvalues2, next_done, gamma, gae_lambda):
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    # TODO: optimize this
    done_used1 = torch.ones_like(next_done, dtype=torch.bool)
    done_used2 = torch.ones_like(next_done, dtype=torch.bool)
    reward1 = reward2 = 0
    lastgaelam1 = lastgaelam2 = 0
    for t in reversed(range(num_steps)):
        # if learns[t]:
        #     if dones[t+1]:
        #         reward1 = rewards[t]
        #         nextvalues1 = 0
        #         lastgaelam1 = 0
        #         done_used1 = True
        #
        #         reward2 = -rewards[t]
        #         done_used2 = False
        #     else:
        #         if not done_used1:
        #             reward1 = reward1
        #             nextvalues1 = 0
        #             lastgaelam1 = 0
        #             done_used1 = True
        #         else:
        #             reward1 = rewards[t]
        #         reward2 = reward2
        #     delta1 = reward1 + args.gamma * nextvalues1 - values[t]
        #     lastgaelam1_ = delta1 + args.gamma * args.gae_lambda * lastgaelam1
        #     advantages[t] = lastgaelam1_
        #     nextvalues1 = values[t]
        #     lastgaelam1 = lastgaelam_
        # else:
        #     if dones[t+1]:
        #         reward2 = rewards[t]
        #         nextvalues2 = 0
        #         lastgaelam2 = 0
        #         done_used2 = True
        #
        #         reward1 = -rewards[t]
        #         done_used1 = False
        #     else:
        #         if not done_used2:
        #             reward2 = reward2
        #             nextvalues2 = 0
        #             lastgaelam2 = 0
        #             done_used2 = True
        #         else:
        #             reward2 = rewards[t]
        #         reward1 = reward1
        #     delta2 = reward2 + args.gamma * nextvalues2 - values[t]
        #     lastgaelam2_ = delta2 + args.gamma * args.gae_lambda * lastgaelam2
        #     advantages[t] = lastgaelam2_
        #     nextvalues2 = values[t]
        #     lastgaelam2 = lastgaelam_

        learn1 = learns[t]
        learn2 = ~learn1
        if t != num_steps - 1:
            next_done = dones[t + 1]
        sp = 2 * (learn1.int() - 0.5)
        reward1 = torch.where(next_done, rewards[t] * sp, torch.where(learn1 & done_used1, 0, reward1))
        reward2 = torch.where(next_done, rewards[t] * -sp, torch.where(learn2 & done_used2, 0, reward2))
        real_done1 = next_done | ~done_used1
        nextvalues1 = torch.where(real_done1, 0, nextvalues1)
        lastgaelam1 = torch.where(real_done1, 0, lastgaelam1)
        real_done2 = next_done | ~done_used2
        nextvalues2 = torch.where(real_done2, 0, nextvalues2)
        lastgaelam2 = torch.where(real_done2, 0, lastgaelam2)
        done_used1 = torch.where(
            next_done, learn1, torch.where(learn1 & ~done_used1, True, done_used1))
        done_used2 = torch.where(
            next_done, learn2, torch.where(learn2 & ~done_used2, True, done_used2))

        delta1 = reward1 + gamma * nextvalues1 - values[t]
        delta2 = reward2 + gamma * nextvalues2 - values[t]
        lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
        lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
        advantages[t] = torch.where(learn1, lastgaelam1_, lastgaelam2_)
        nextvalues1 = torch.where(learn1, values[t], nextvalues1)
        nextvalues2 = torch.where(learn2, values[t], nextvalues2)
        lastgaelam1 = torch.where(learn1, lastgaelam1_, lastgaelam1)
        lastgaelam2 = torch.where(learn2, lastgaelam2_, lastgaelam2)
    return advantages


def bootstrap_value_selfplay_upgo(values, rewards, dones, learns, nextvalues1, nextvalues2, next_done, gamma, gae_lambda):
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    # TODO: optimize this
    done_used1 = torch.ones_like(next_done, dtype=torch.bool)
    done_used2 = torch.ones_like(next_done, dtype=torch.bool)
    reward1 = reward2 = 0
    lastgaelam1 = lastgaelam2 = 0
    for t in reversed(range(num_steps)):
        # if learns[t]:
        #     if dones[t+1]:
        #         reward1 = rewards[t]
        #         next_values1 = 0
        #         last_return1 = 0
        #         lastgaelam1 = 0
        #         done_used1 = True
        #
        #         reward2 = -rewards[t]
        #         done_used2 = False
        #     else:
        #         if not done_used1:
        #             reward1 = reward1
        #             next_values1 = 0
        #             last_return1 = 0
        #             lastgaelam1 = 0
        #             done_used1 = True
        #         else:
        #             reward1 = rewards[t]
        #         reward2 = reward2
        #     last_return1_ = reward1 + args.gamma * (last_return1 if (next_qs1 >= next_values1) else next_values1)
        #     next_q1_ = reward1 + args.gamma * next_values1
        #     delta1 = next_q1_ - values[t]
        #     lastgaelam1_ = delta1 + args.gamma * args.gae_lambda * lastgaelam1
        #     returns[t] = last_return1_
        #     advantages[t] = lastgaelam1_
        #     next_values1 = values[t]
        #     lastgaelam1 = lastgaelam1_
        #     next_qs1 = next_q1_
        #     last_return1 = last_return1_
        # else:
        #     Skip because it is symmetric

        learn1 = learns[t]
        learn2 = ~learn1
        if t != num_steps - 1:
            next_done = dones[t + 1]
        sp = 2 * (learn1.int() - 0.5)
        reward1 = torch.where(next_done, rewards[t] * sp, torch.where(learn1 & done_used1, 0, reward1))
        reward2 = torch.where(next_done, rewards[t] * -sp, torch.where(learn2 & done_used2, 0, reward2))
        real_done1 = next_done | ~done_used1
        nextvalues1 = torch.where(real_done1, 0, nextvalues1)
        lastgaelam1 = torch.where(real_done1, 0, lastgaelam1)
        real_done2 = next_done | ~done_used2
        nextvalues2 = torch.where(real_done2, 0, nextvalues2)
        lastgaelam2 = torch.where(real_done2, 0, lastgaelam2)
        done_used1 = torch.where(
            next_done, learn1, torch.where(learn1 & ~done_used1, True, done_used1))
        done_used2 = torch.where(
            next_done, learn2, torch.where(learn2 & ~done_used2, True, done_used2))

        delta1 = reward1 + gamma * nextvalues1 - values[t]
        delta2 = reward2 + gamma * nextvalues2 - values[t]
        lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
        lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
        advantages[t] = torch.where(learn1, lastgaelam1_, lastgaelam2_)
        nextvalues1 = torch.where(learn1, values[t], nextvalues1)
        nextvalues2 = torch.where(learn2, values[t], nextvalues2)
        lastgaelam1 = torch.where(learn1, lastgaelam1_, lastgaelam1)
        lastgaelam2 = torch.where(learn2, lastgaelam2_, lastgaelam2)
    return advantages


def bootstrap_value_selfplay_np(values, rewards, dones, learns, nextvalues1, nextvalues2, next_done, gamma, gae_lambda):
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    # TODO: optimize this
    done_used1 = np.ones_like(next_done, dtype=np.bool_)
    done_used2 = np.ones_like(next_done, dtype=np.bool_)
    reward1 = reward2 = 0
    lastgaelam1 = lastgaelam2 = 0
    for t in reversed(range(num_steps)):
        learn1 = learns[t]
        learn2 = ~learn1
        if t != num_steps - 1:
            next_done = dones[t + 1]
        sp = 2 * (learn1.astype(np.float32) - 0.5)
        reward1 = np.where(next_done, rewards[t] * sp, np.where(learn1 & done_used1, 0, reward1))
        reward2 = np.where(next_done, rewards[t] * -sp, np.where(learn2 & done_used2, 0, reward2))
        real_done1 = next_done | ~done_used1
        nextvalues1 = np.where(real_done1, 0, nextvalues1)
        lastgaelam1 = np.where(real_done1, 0, lastgaelam1)
        real_done2 = next_done | ~done_used2
        nextvalues2 = np.where(real_done2, 0, nextvalues2)
        lastgaelam2 = np.where(real_done2, 0, lastgaelam2)
        done_used1 = np.where(
            next_done, learn1, np.where(learn1 & ~done_used1, True, done_used1))
        done_used2 = np.where(
            next_done, learn2, np.where(learn2 & ~done_used2, True, done_used2))

        delta1 = reward1 + gamma * nextvalues1 - values[t]
        delta2 = reward2 + gamma * nextvalues2 - values[t]
        lastgaelam1_ = delta1 + gamma * gae_lambda * lastgaelam1
        lastgaelam2_ = delta2 + gamma * gae_lambda * lastgaelam2
        advantages[t] = np.where(learn1, lastgaelam1_, lastgaelam2_)
        nextvalues1 = np.where(learn1, values[t], nextvalues1)
        nextvalues2 = np.where(learn2, values[t], nextvalues2)
        lastgaelam1 = np.where(learn1, lastgaelam1_, lastgaelam1)
        lastgaelam2 = np.where(learn2, lastgaelam2_, lastgaelam2)
    return advantages