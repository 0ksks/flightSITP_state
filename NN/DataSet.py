from torch.utils.data import Dataset

from InfoPrinter import heading
from DataLoader import split_dataloader as _split
from DataLoader import mini_dataloader as _mini
from typing import Union
import pickle


class __DatasetName(Dataset):
    def __init__(self, params: dict = None) -> None:
        with open(params["filepath"], "rb") as f:
            data = pickle.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def test(self) -> None:
        heading("data size")
        print(len(self))
        heading("samples")
        for idx, sample in enumerate(self.data[0:3]):
            print(f"sample {idx}:\n{sample}")

    @property
    def shape(self) -> tuple:
        shape_ = [0, 0]
        for item in self.data:
            shape = item["shape"]
            if shape[0] > shape_[0]:
                shape_[0] = shape[0]
            if shape[1] > shape_[1]:
                shape_[1] = shape[1]
        return tuple(shape_)

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
    datasetName = get_dataset({"filepath": "data/NNSETs/data.pickle"})
    datasetName.test()
    loaders = datasetName.split_loader()
    print(datasetName.shape)
