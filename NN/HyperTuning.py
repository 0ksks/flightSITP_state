from NnTrain import kernel
import optuna


def objective_lr(trial: optuna.trial.Trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batchSize = 128
    train_loss = kernel(lr, batchSize)
    return train_loss


"""
optuna-dashboard link
"""

if __name__ == "__main__":
    with open("storageLink.txt", "r") as f:
        link = f.read()
    study_lr = optuna.create_study(
        storage=link, study_name="STUDY_NAME", load_if_exists=True
    )
    study_lr.optimize(objective_lr, n_trials=100)
    print(f"Best value: {study_lr.best_value} (params: {study_lr.best_params})")
