import torch
from typing import List


def make_device(gpu_ids: List[int]) -> torch.device:
    # use by default if none specified
    if not gpu_ids and torch.cuda.is_available():
        gpu_ids = [0]
    # negative index means no gpu
    if not gpu_ids or gpu_ids[0] < 0:
        return torch.device("cpu")
    # use gpu
    return torch.device("cuda:{}".format(gpu_ids[0]))
