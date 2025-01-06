import torch
from torch.utils.data import DistributedSampler, Dataset
from typing import Optional, Iterator, TypeVar

T_co = TypeVar('T_co', covariant=True)

###
# This class is a custom implementation of the DistributedSampler class from PyTorch
# It uses the weights of the samples to sample the data in a distributed manner


class DistributedWeightedSampler(DistributedSampler):
    
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank=None, shuffle=True, seed=0, drop_last=False, weights=None, replace=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.weights = weights
        self.replace = replace

    def __iter__(self) -> Iterator[T_co]:
        generator = None

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            weights = self.weights[indices]
        else:
            indices = list(range(len(self.dataset)))
            weights = self.weights 

        if not self.drop_last:
            raise NotImplementedError('Drop last is not implemented')
        else:
            weights = weights[:self.total_size]

        weights_split = weights[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        rand_indices = torch.multinomial(weights_split, len(weights_split), generator=generator, replacement=self.replace)
        sampled_indices = torch.zeros_like(rand_indices, dtype=torch.int32)
        
        for i, index in enumerate(rand_indices):
            sampled_indices[i] = indices[index.item()]

        yield from iter(sampled_indices.tolist())