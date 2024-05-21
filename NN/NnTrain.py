def kernel(hyperParams=None):
    from DataSet import get_dataset
    from NnStructure import FlightTransformer
    import pytorch_lightning as pl

    model = FlightTransformer()
    trainer = pl.Trainer()
    loaders = get_dataset().split_loader()
    trainer.fit(model, loaders[0])
    train_loss = trainer.callback_metrics["train_loss"].item()
    return train_loss


if __name__ == "__main__":
    kernel()
