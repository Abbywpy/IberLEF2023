import yaml

default_config = {
    'model': {
        'hidden_size': 256,
        'dropout_rate': 0.15,
        'num_layers': 2,
        'bias': False,
        'lr': 1e-3,

    },
    'max_length': 128,

}

with open('simpleCLF_config.yaml', 'w') as f:
    yaml.dump(default_config, f)
