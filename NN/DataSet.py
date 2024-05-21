from torch.utils.data import Dataset
import numpy as np
from InfoPrinter import heading
from DataLoader import split_dataloader as _split
from DataLoader import mini_dataloader as _mini
from typing import Union


class __DatasetName(Dataset):
    def __init__(self, params: dict = None) -> None:
        self.data = np.random.randint(1, 10, (10000, 2))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        
        return self.data.__getitem__(index)

    def test(self) -> None:
        heading("data size")
        print(len(self))
        heading("samples")
        print(self[0:3])

    def split_loader(
        self,
        ratio: tuple = (8, 1, 1),
        batchSize: Union[int, list[int]] = 10,
        shuffle=True,
        randomSeed=2003,
        mini=False,
        subSize=100,
        subBatchSize=10,
        subShuffle=True,
    ):
        dataloaders = _split(self, ratio, batchSize, shuffle, randomSeed)
        if mini:
            for i, dataloader in enumerate(dataloaders):
                if i == 0:
                    dataloaders[i] = _mini(
                        dataloader, subSize, subBatchSize, subShuffle
                    )
                else:
                    dataloaders[i] = _mini(dataloader, subSize, subBatchSize, False)
        return dataloaders


def get_dataset(params: dict = None) -> __DatasetName:
    __datasetName = __DatasetName(params)
    return __datasetName


if __name__ == "__main__":
    datasetName = get_dataset()
    datasetName.test()
    loaders = datasetName.split_loader()
