from torch.utils.data import random_split, DataLoader, Subset, Dataset
import torch
from typing import Union


def split_dataloader(
    dataset: Dataset,
    ratio: tuple,
    batchSize: Union[int, list[int]],
    shuffle: bool,
    randomSeed,
) -> list[DataLoader]:
    if isinstance(batchSize, int):
        batchSize = [
            batchSize,
        ] * 3
    ratio = list(map(lambda x: x / sum(ratio), ratio))
    ratio[-1] = 1 - sum(ratio[:-1])
    print(f"split by {' : '.join(map(lambda x:f'{100*x:.1f}%',ratio))}")
    generator = torch.Generator().manual_seed(randomSeed)
    datasets = random_split(dataset, ratio, generator)
    dataloaders = []
    for idx, dataset in enumerate(datasets):
        if idx == 0:
            dataloaders.append(DataLoader(dataset, batchSize[idx], shuffle))
        else:
            dataloaders.append(DataLoader(dataset, batchSize[idx]))
    return dataloaders


def mini_dataloader(
    dataloader: DataLoader, subSize: int, subBatchSize: int, subShuffle: bool
):
    oriDataset = dataloader.dataset
    oriDataSize = len(oriDataset)
    import random

    randomIndices = random.sample(range(oriDataSize), subSize)
    return DataLoader(
        Subset(oriDataset, randomIndices), subBatchSize, shuffle=subShuffle
    )
