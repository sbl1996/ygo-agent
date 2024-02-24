import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reduce_gradidents(model, world_size):
    if world_size == 1:
        return
    all_grads_list = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads_list.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.copy_(
                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
            )
            offset += param.numel()


def setup(backend, rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # manual init nccl
    x = torch.rand(4, device=f'cuda:{rank}')
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.mean().item()
    dist.barrier()
    # print(f"Rank {rank} initialized")


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
