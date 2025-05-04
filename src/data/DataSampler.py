import numpy as np
from torch.utils.data.distributed import DistributedSampler
from collections import Counter, defaultdict
import math
import torch.distributed as dist
import random
from torch.utils.data import Dataset

class MyDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, mode='up', num_replicas=None, rank=None, shuffle=True, seed=42, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.labels = self.dataset.get_labels()
        self.mode = mode  # mode in ('up', 'down', 'mid')
        self.label_index = defaultdict(list)
        self.indices = []
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # balance smaple
        cnt = Counter(self.labels)
        for i, label in enumerate(self.labels):
            self.label_index[label].append(i)
        if self.mode == 'mid':
            median_cnt = int(np.median(list(cnt.values())))
            self.__midsample_indices(median_cnt)
        elif self.mode == 'up':
            max_cnt = max(cnt.values())
            self.__upsample_indices(max_cnt)
        elif self.mode == 'down':
            min_cnt = min(cnt.values())
            self.__downsample_indices(min_cnt)
        else:
            raise NotImplementedError

        if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __upsample_indices(self, up_num):
        for label in self.label_index:
            indices = self.label_index[label]
            if len(indices) < up_num:
                indices = np.random.choice(indices, up_num, replace=True).tolist()
            self.indices.extend(indices)

    def __midsample_indices(self, mid_num):
        for label in self.label_index:
            indices = self.label_index[label]
            if len(indices) < mid_num:
                indices = np.random.choice(indices, mid_num, replace=True).tolist()
            elif len(indices) > mid_num:
                indices = np.random.choice(indices, mid_num).tolist()
            self.indices.extend(indices)

    def __downsample_indices(self, down_num):
        for label in self.label_index:
            indices = self.label_index[label]
            if len(indices) > down_num:
                indices = np.random.choice(indices, down_num).tolist()
            self.indices.extend(indices)

    # copy for torch.utils.data.distributed.DistributedSampler
    def __iter__(self):
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
        indices = self.indices

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch