import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, device)


def wrap_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if context.world_size == 1:
        return model
    if context.device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[context.local_rank])
    return DistributedDataParallel(model)


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= context.world_size
    return result


def synchronize(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def finalize_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
