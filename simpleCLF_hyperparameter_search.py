import optuna
import torch
import lightning as L
from trainer import SpanishTweetsCLF
from dataloader import SpanishTweetsDataModule
import yaml

def objective(trial: optuna.trial.Trial) -> float:
    # search space for hyperparameters
    lr = trial.suggest_categorical("learning_rate", [1e-3, 2e-5, 3e-5])
    dropout = trial.suggest_float("dropout", 0.1, 0.2, step=0.05)
    epochs = trial.suggest_int("epochs", 5, 20, step=1)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 768])
    num_layers = trial.suggest_int("num_layers", 1, 2, step=1)
    #batch_size = trial.suggest_int("batch_size", 16, 62, step=16)

    model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=lr, dropout_rate=dropout, hidden_size=hidden_size, num_layers=num_layers, bias=False)

    dm = SpanishTweetsDataModule(
            train_dataset_path="data/hp_data/hp_cleaned_encoded_train.csv", # path leads to subset of full data especially created for hp search
            val_dataset_path="data/hp_data/hp_cleaned_encoded_development.csv", # path leads to subset of full data especially created for hp search
            batch_size=8,
            num_workers=8)

    # Create the Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu")
    
    trainer.fit(model, dm)

    avg_precision = sum([trainer.logged_metrics[f"valid_{attr}_precision"] for attr in model.attr]) / len(model.attr)
    avg_recall = sum([trainer.logged_metrics[f"valid_{attr}_recall"] for attr in model.attr]) / len(model.attr)
    avg_f1 = sum([trainer.logged_metrics[f"valid_{attr}_f1"] for attr in model.attr]) / len(model.attr)

    final_metric = (avg_precision + avg_recall + avg_f1) / 3

    return final_metric


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="SpanishTweetsCLF_gender", load_if_exists=True)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trials = study.get_trials()

    with open("gender_trial_results.txt", "w") as file:
        for trial in trials:
            trial_number = trial.number
            trial_value = trial.value
            trial_params = trial.params

            file.write(f"Trial Number: {trial_number}\n")
            file.write(f"Trial Value: {trial_value}\n")
            file.write("Trial Parameters:\n")
            for param_name, param_value in trial_params.items():
                file.write(f"  {param_name}: {param_value}\n")
            file.write("\n")

    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_hyperparams = trial.params
    
    with open("best_hyperparams_gender.yaml", "w") as f:
        yaml.dump(best_hyperparams, f)
        
        print(f"The best hyperparameters are saved in 'best_hyperparams_gender.yaml'." )
    
    
    
