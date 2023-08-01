args_wideresnet = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 256,
    'num_workers': 12,
}
args_preactresnet18 = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 256,
    'num_workers': 12,
}

args_resnet50 = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 256,
    'num_workers': 12,
}

args_resnext = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 32,
    'num_workers': 12,
}

args_vgg16 = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 256,
    'num_workers': 12,
}

args_vgg19 = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 256,
    'num_workers': 12,
}

args_densenet121 = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 128,
    'num_workers': 12,
}

args_senet = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 100
    },
    'batch_size': 32,
    'num_workers': 12,
}