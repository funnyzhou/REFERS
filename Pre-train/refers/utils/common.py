import argparse
import os
import random
import sys

from loguru import logger
import numpy as np
import torch

from refers.config import Config
import refers.utils.distributed as dist
import pdb


def cycle(dataloader, device, start_iteration: int = 0):
    r"""
    A generator to yield batches of data from dataloader infinitely.

    Internally, it sets the ``epoch`` for dataloader sampler to shuffle the
    examples. One may optionally provide the starting iteration to make sure
    the shuffling seed is different and continues naturally.
    """
    iteration = start_iteration

    while True:
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            # Set the `epoch` of DistributedSampler as current iteration. This
            # is a way of determinisitic shuffling after every epoch, so it is
            # just a seed and need not necessarily be the "epoch".
            logger.info(f"Beginning new epoch, setting shuffle seed {iteration}")
            dataloader.sampler.set_epoch(iteration)

        for batch in dataloader:
            for key in batch:
                # pdb.set_trace()
                if key == 'contrastive_caption':
                    batch[key]['input_ids'] = batch[key]['input_ids'].to(device)
                    batch[key]['token_type_ids'] = batch[key]['token_type_ids'].to(device)
                    batch[key]['attention_mask'] = batch[key]['attention_mask'].to(device)
                else:
                    batch[key] = batch[key].to(device)
            yield batch
            iteration += 1


def common_setup(_C: Config, _A: argparse.Namespace, job_type: str = "pretrain"):
    r"""
    Setup common stuff at the start of every pretraining or downstream
    evaluation job, all listed here to avoid code duplication. Basic steps:

    1. Fix random seeds and other PyTorch flags.
    2. Set up a serialization directory and loggers.
    3. Log important stuff such as config, process info (useful during
        distributed training).
    4. Save a copy of config to serialization directory.

    .. note::

        It is assumed that multiple processes for distributed training have
        already been launched from outside. Functions from
        :mod:`refers.utils.distributed` module ae used to get process info.

    Parameters
    ----------
    _C: refers.config.Config
        Config object with all the parameters.
    _A: argparse.Namespace
        Command line arguments.
    job_type: str, optional (default = "pretrain")
        Type of job for which setup is to be done; one of ``{"pretrain",
        "downstream"}``.
    """

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(_C.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, f"{job_type}_config.yaml"))

    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
    if dist.get_world_size() > 1:
        logger.add(
            os.path.join(_A.serialization_dir, f"log-rank{RANK}.txt"),
            format="{time} {level} {message}",
        )

    # Add a logger for stdout only for the master process.
    if dist.is_master_process():
        logger.add(
            sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
        )

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(str(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))
