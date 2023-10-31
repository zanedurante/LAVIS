"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import functools
import os

import torch
import torch.distributed as dist
import timm.models.hub as timm_hub


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_dist_args_gcr():
    envvars = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "NODE_COUNT",
        "HOSTNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "AZ_BATCHAI_MPI_MASTER_NODE",
    ]
    args = dict(gpus_per_node=torch.cuda.device_count())
    missing = []
    for var in envvars:
        if var in os.environ:
            args[var] = os.environ.get(var)
            try:
                args[var] = int(args[var])
            except ValueError:
                pass
        else:
            missing.append(var)
    print(f"II Args: {args}")
    if missing:
        print(f"II Environment variables not set: {', '.join(missing)}.")
    return args


def init_distributed_gcr(): # Special distributed setup for HAI clusters on GCR
    gpus_per_node = torch.cuda.device_count()
    dist_args = get_dist_args_gcr()
    world_size = dist_args.get("WORLD_SIZE", 1)
    node_rank = dist_args.get("NODE_RANK", 0)
    local_rank = dist_args.get("LOCAL_RANK", 0)
    gpu_rank = local_rank % gpus_per_node
    master_addr = dist_args.get("MASTER_ADDR", "localhost")
    master_port = dist_args.get("MASTER_PORT", "12323")
    print(f"WORLD_SIZE: {world_size}, GPU_RANK: {gpu_rank}, NODE RANK: {node_rank}\n")
    # Add support for single gpu dist training with same launcher
    if node_rank is None:
        node_rank = 0 
    if world_size is None:
        world_size = 1
    if gpu_rank is None:
        gpu_rank = 0
    if master_addr is None:
        os.environ['MASTER_ADDR'] = 'localhost'
    if master_port is None:
        os.environ['MASTER_PORT'] = '12323'

    if node_rank > 0:
        os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        master_addr = os.environ['MASTER_IP'] # Master IP set separately
    
    global_rank = node_rank * gpus_per_node + gpu_rank
    master_uri = "tcp://%s:%s" % (master_addr, master_port)
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    print(f"WORLD_SIZE: {world_size}, GPU_RANK: {gpu_rank}, NODE RANK: {node_rank} MASTER_ADDR: {master_addr} \n")
    
    if "None" in master_uri: # Use standard init if not set
        dist.init_process_group(
            backend='nccl', rank=global_rank, world_size=world_size
        )
    else:
        dist.init_process_group(
            backend='nccl', rank=global_rank, init_method=master_uri, world_size=world_size
        )
    torch.cuda.set_device(gpu_rank)
    print(f"II: Rank {global_rank} initialized.")
    torch.distributed.barrier()
    setup_for_distributed(global_rank == 0)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()
