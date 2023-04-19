import optuna
import torch
import lightning as L
from trainer import SpanishTweetsCLF
from dataloader import SpanishTweetsDataModule
import yaml

def objective(trial: optuna.trial.Trial) -> float:
    # search space for hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 2, step=1)
    #batch_size = trial.suggest_int("batch_size", 16, 62, step=16)

   # lr = trial.suggest_categorical("learning_rate", [1e-3, 2e-5, 3e-5])
    #dropout = trial.suggest_float("dropout", 0.1, 0.2, step=0.05)
    #epochs = trial.suggest_int("epochs", 5, 10, step=1)
    #hidden_size = trial.suggest_int("hidden_size", 128, 768, step=128)
    #num_layers = trial.suggest_int("num_layers", 1, 2, step=1)
    #batch_size = trial.suggest_int("batch_size", 16, 62, step=16)

    #model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=lr, dropout_rate=dropout, hidden_size=hidden_size, num_layers=num_layers, bias=False)
    
    model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=1e-3, dropout_rate=0.2, hidden_size=128, num_layers=num_layers, bias=False)

    # TODO: change path to larger data set for actual hp search
    # "data/hp_data/hp_cleaned_encoded_train.csv"
    # "data/hp_data/hp_cleaned_encoded_train.csv"
    dm = SpanishTweetsDataModule(
            train_dataset_path="data/practise_data/cleaned/tiny_cleaned_encoded_train.csv", # path leads to *very* small subset of practise data
            val_dataset_path="data/practise_data/cleaned/tiny_cleaned_encoded_development.csv", # path leads to *very* small subset of practise data
            batch_size=2)

    # Create the Trainer
    # TODO: Add "gpus=1" argument for gpu usage
    trainer = L.Trainer(
        max_epochs=5, # TODO: change back to "epochs"
    )
    
    trainer.fit(model, dm)

    avg_precision = sum([trainer.logged_metrics[f"valid_{attr}_precision"] for attr in model.attr]) / len(model.attr)
    avg_recall = sum([trainer.logged_metrics[f"valid_{attr}_recall"] for attr in model.attr]) / len(model.attr)
    avg_f1 = sum([trainer.logged_metrics[f"valid_{attr}_f1"] for attr in model.attr]) / len(model.attr)

    final_metric = (avg_precision + avg_recall + avg_f1) / 3

    return final_metric


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="SpanishTweetsCLF")
    study.optimize(objective, n_trials=1) # TODO: change n_trials to 10

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    best_hyperparams = trial.params
    
    with open("best_hyperparams.yaml", "w") as f:
        yaml.dump(best_hyperparams, f)
        
        print(f"The best hyperparameters are saved in 'best_hyperparams_simpleCLF.yaml'." )
    
    
    