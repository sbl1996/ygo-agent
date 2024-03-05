import os
import sys
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reduce_gradidents(params, world_size):
    if world_size == 1:
        return
    all_grads_list = []
    for param in params:
        if param.grad is not None:
            all_grads_list.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in params:
        if param.grad is not None:
            param.grad.data.copy_(
                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
            )
            offset += param.numel()


def test_nccl(local_rank):
    # manual init nccl
    x = torch.rand(4, device=f'cuda:{local_rank}')
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.mean().item()
    dist.barrier()


def torchrun_setup(backend, local_rank):
    dist.init_process_group(
        backend, timeout=datetime.timedelta(seconds=60 * 30))
    test_nccl(local_rank)


def setup(backend, rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(
        backend, rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=60 * 30))

    test_nccl(rank)


def mp_start(run):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        run(local_rank=0, world_size=world_size)
    else:
        # mp.set_start_method('spawn')
        children = []
        for i in range(world_size):
            subproc = mp.Process(target=run, args=(i, world_size))
            children.append(subproc)
            subproc.start()

        for i in range(world_size):
            children[i].join()


def fprint(msg):
    sys.stdout.flush()
    sys.stdout.write(msg + os.linesep)
    sys.stdout.flush()
