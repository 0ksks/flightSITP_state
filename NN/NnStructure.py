import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


def padding(srcMat: torch.Tensor, tgtShape: tuple[int]) -> torch.Tensor:
    import warnings

    srcShape = srcMat.shape
    top = 0
    bottom = tgtShape[0] - srcShape[0]
    left = 0
    right = tgtShape[1] - srcShape[1]
    if bottom < 0 or right < 0:
        warnings.warn(
            f"the target shape {tgtShape} is smaller than the source shape {tuple(srcShape)}"
        )
    return F.pad(srcMat, (left, right, top, bottom), mode="constant", value=0)


class FlightTransformer(pl.LightningModule):
    def __init__(self, optimParams=None, optim_=optim.Adam) -> None:
        super(FlightTransformer, self).__init__()
        self.optim = optim_
        self.optimParams = optimParams

    def forward(self, input_):
        return input_

    def training_step(self, batch, batch_idx):
        self.log("train_loss", 0)
        return

    def configure_optimizers(self):
        if self.optimParams is not None:
            optimizer = self.optim(self.parameters(), self.optimParams)
        else:
            optimizer = self.optim(self.parameters())
        return optimizer


if __name__ == "__main__":
    inputTensor = torch.randn((2, 3))
    model = FlightTransformer()
    outputTensor = model(inputTensor)
    print(f"{inputTensor = }")
    print(f"{outputTensor = }")
