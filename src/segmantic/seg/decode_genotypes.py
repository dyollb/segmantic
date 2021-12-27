def decode_optimizer_gene(code):
    if code == 0:
        return {'optimizer': 'SGD',
                'lr': 1e-3,
                'momentum': 0.8,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 1:
        return {'optimizer': 'SGD',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 2:
        return {'optimizer': 'SGD',
                'lr': 1e-4,
                'momentum': 0.8,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 3:
        return {'optimizer': 'SGD',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 4:
        return {'optimizer': 'Adam',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 5:
        return {'optimizer': 'Adam',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': True,
                'weight_decouple': False}
    elif code == 6:
        return {'optimizer': 'Adam',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 7:
        return {'optimizer': 'Adam',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': True,
                'weight_decouple': False}
    elif code == 8:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 9:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': True}
    elif code == 10:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-16,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 11:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-3,
                'momentum': 0.9,
                'epsilon': 1e-16,
                'amsgrad': False,
                'weight_decouple': True}
    elif code == 12:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 13:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-8,
                'amsgrad': False,
                'weight_decouple': True}
    elif code == 14:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-16,
                'amsgrad': False,
                'weight_decouple': False}
    elif code == 15:
        return {'optimizer': 'AdaBelief',
                'lr': 1e-4,
                'momentum': 0.9,
                'epsilon': 1e-16,
                'amsgrad': False,
                'weight_decouple': True}


def decode_lr_scheduler_gene(code):
    if code == 0:
        return {'scheduler': 'Constant',
                'factor': 0.5,
                'patience': 10,
                'T_0': 50,
                'T_multi': 1}
    elif code == 1:
        return {'scheduler': 'ReduceOnPlateau',
                'factor': 0.5,
                'patience': 10,
                'T_0': 50,
                'T_multi': 1}
    elif code == 2:
        return {'scheduler': 'ReduceOnPlateau',
                'factor': 0.5,
                'patience': 20,
                'T_0': 50,
                'T_multi': 1}
    elif code == 3:
        return {'scheduler': 'ReduceOnPlateau',
                'factor': 0.1,
                'patience': 10,
                'T_0': 50,
                'T_multi': 1}
    elif code == 4:
        return {'scheduler': 'ReduceOnPlateau',
                'factor': 0.1,
                'patience': 20,
                'T_0': 50,
                'T_multi': 1}
    elif code == 5:
        return {'scheduler': 'Cosine',
                'factor': 0.5,
                'patience': 10,
                'T_0': 50,
                'T_multi': 1}
    elif code == 6:
        return {'scheduler': 'Cosine',
                'factor': 0.5,
                'patience': 10,
                'T_0': 1,
                'T_multi': 2}
    elif code == 7:
        return {'scheduler': 'Cosine',
                'factor': 0.5,
                'patience': 10,
                'T_0': 10,
                'T_multi': 2}
