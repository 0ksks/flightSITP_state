import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pprint import pp
from typing import Union
from scipy.sparse import coo_matrix
from numpy import array, where, ndarray
from InfoPrinter import heading
from DataLoader import split_dataloader as _split
from DataLoader import mini_dataloader as _mini


def matrix_transformer(item: dict) -> dict:
    itemInput = coo_matrix(item["input"], item["shape"]).todense()
    itemInput = array(itemInput)
    item["input"] = itemInput
    return item


def pad_matrix(item: dict, tgtShape: tuple[int]) -> dict:

    itemInput = item["input"]
    itemInput = F.pad(
        torch.tensor(itemInput),
        (
            0,
            tgtShape[1] - item["shape"][1],
            0,
            tgtShape[0] - item["shape"][0],
        ),
        value=-1,
    )
    item["input"] = itemInput
    return item


def pad_list(input_: list, tgtShape: int) -> torch.Tensor:
    input_ = torch.tensor(input_)
    return F.pad(input_, (0, tgtShape), value=-1)


def normal(data: dict) -> dict:
    data["cancel"] = array(data["cancel"])
    max_ = max(data["input"].max(), data["cancel"].max(), data["cost"])
    data["input"] /= max_
    data["cancel"] /= max_
    data["cost"] /= max_
    return data


class __DatasetName(Dataset):
    def __init__(self, params: dict = None) -> None:
        with open(params["filepath"], "rb") as f:
            data: list[dict] = pickle.load(f)
        self.data = data
        shape_ = [0, 0]
        for item in self.data:
            shape = item["shape"]
            if shape[0] > shape_[0]:
                shape_[0] = shape[0]
            if shape[1] > shape_[1]:
                shape_[1] = shape[1]
        self.shape = tuple(shape_)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        def pipeline(item: dict) -> dict:
            item = matrix_transformer(item)
            item = normal(item)
            item = pad_matrix(item, self.shape)
            item["shape"] = torch.tensor(item["shape"], dtype=int)
            item["pos"] = pad_list(item["pos"], self.shape[1] - item["shape"][1])
            item["cancel"] = pad_list(item["cancel"], self.shape[1] - item["shape"][1])
            item["output"] = pad_list(item["output"], self.shape[0] - item["shape"][0])
            item["cost"] = torch.tensor(item["cost"])
            return item

        data = self.data.__getitem__(index)
        if isinstance(data, list):
            return list(map(pipeline, data))
        elif isinstance(data, dict):
            return pipeline(data)

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

    loaders = datasetName.split_loader()
    loader = loaders[0]
    for idx, item in enumerate(loader):
        print(idx, item)
        if idx == 10:
            break
