import re

import optree
import torch

from ygoai.rl.env import RecordEpisodeStatistics, EnvPreprocess
    

def split_param_groups(model, regex):
    embed_params = []
    other_params = []
    for name, param in model.named_parameters():
        if re.search(regex, name):
            embed_params.append(param)
        else:
            other_params.append(param)
    return [
        {'params': embed_params}, {'params': other_params}
    ]


class Elo:

    def __init__(self, k = 10, r0 = 1500, r1 = 1500):
        self.r0 = r0
        self.r1 = r1
        self.k = k

    def update(self, winner):
        diff = self.k * (1 - self.expect_result(self.r0, self.r1))
        if winner == 1:
            diff = -diff
        self.r0 += diff
        self.r1 -= diff

    def expect_result(self, p0, p1):
        exp = (p0 - p1) / 400.0
        return 1 / ((10.0 ** (exp)) + 1)
    

def masked_mean(x, valid):
    x = x.masked_fill(~valid, 0)
    return x.sum() / valid.float().sum()


def masked_normalize(x, valid, eps=1e-8):
    x = x.masked_fill(~valid, 0)
    n = valid.float().sum()
    mean = x.sum() / n
    var = ((x - mean) ** 2).sum() / n
    std = (var + eps).sqrt()
    return (x - mean) / std


def to_tensor(x, device, dtype=None):
    return optree.tree_map(lambda x: torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True), x)

