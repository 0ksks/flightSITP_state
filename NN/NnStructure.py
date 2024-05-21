import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class ModelName(pl.LightningModule):
    def __init__(self, optimParams=None, optim_=optim.Adam) -> None:
        super(ModelName, self).__init__()
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
    model = ModelName()
    outputTensor = model(inputTensor)
    print(f"{inputTensor = }")
    print(f"{outputTensor = }")
