from utils import get_now

config = {
    'epochs': 12,
    'lr' : 1e-4,
    'batch_size': 16
}

upper_class_name ={
    1 : 'texture',
    2 : 'fabric',
    3 : 'shape',
    4 : 'part',
    5 : 'style'
}

sweep_configuration = {
    'method': 'bayes',
    'name': f'TSFashionNet_sweep_{get_now(True)}',
    'project': 'tsfashionnet',
    'entity': 'ljh415',
    'metric': {
        'goal': 'minimize',
        'name': 'valid_loss'
    },
    'parameters': {
        'batch_size': {'values':[16, 32, 64]},
        'epochs': {'min':12, 'max':20},
        'lr': {'min':1e-5, 'max':1e-3},
        'shape_epochs': {'min': 3, 'max': 5},
        'shape_lr': {'min':1e-4, 'max':1e-3},
    }
}