import optuna
from pytorch_lightning import Trainer
from trainer import SpanishTweetsCLF
from dataloader import SpanishTweetsDataModule

def objective(trial: optuna.trial.Trial) -> float:
    # search space for hyperparameters
    lr = trial.suggest_categorical("learning_rate", [1e-3, 2e-5, 3e-5])
    dropout = trial.suggest_float("dropout", 0.1, 0.2, step=0.05)
    epochs = trial.suggest_int("epochs", 5, 10, step=1)
    hidden_size = trial.suggest_int("hidden_size", 128, 768, step=128)
    num_layers = trial.suggest_int("num_layers", 1, 2, step=1)


    model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=lr, dropout_rate=dropout, hidden_size=hidden_size, num_layers=num_layers, bias=False)

    dm = SpanishTweetsDataModule()

    t = Trainer(
        max_epochs=epochs,
        accelerator="cpu",
    )

    # TODO: find out why this is not working
    t.fit(model, datamodule=dm)

    # TODO: implement metric in trainer.py for this to work
    return model.metric


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))