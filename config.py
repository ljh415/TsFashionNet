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
    'vgg': {
        'method': 'bayes',
        'name': f'TSFashionNet_sweep_{get_now(True)}',
        'project': 'tsfashionnet',
        'entity': 'ljh415',
        'metric': {
            'goal': 'minimize',
            'name': 'valid_loss'
        },
        'parameters': {
            # 'batch_size': {'values':[16, 32]},
            'epochs': {'min':12, 'max':20},
            'lr': {'min':1e-5, 'max':3e-3},
            # 'shape_epochs': {'min': 3, 'max': 5},
            # 'shape_lr': {'min':1e-5, 'max':5e-3},
        }
    },
    'bit': {
        'method': 'bayes',
        'name': f'TSFashionNet_sweep_{get_now(True)}',
        'project': 'tsfashionnet',
        'entity': 'ljh415',
        'metric': {
            'goal': 'maximaize',
            'name': 'valid_attribute_recall3'
        },
        'parameters': {
            'cov_epoch': {'values':[1,2,3]},
            'epochs': {'min':12, 'max':20},
            'lr': {'min':3e-3, 'max':1e-2},
            'att_weight': {"min":400, "max":600}
        }
    }
}