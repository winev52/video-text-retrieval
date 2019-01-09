
import itertools
import train_vtt
from constant import CONSTANT
import numpy as np
import os

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def main():
    params = {
        'cpv': [20],
        'margin': np.linspace(0.2, 1, 5),
        'weight_decay': [0, *np.logspace(-8, -16, 3)],
        # 'word_dim': [50, 100, 200, 300],
        # 'word2vec_path': [""],
        'embed_size': np.linspace(256, 1536, 6, dtype=int),
        'learning_rate': np.logspace(-3, -7, 5, dtype=float)
    }

    o_log_path = CONSTANT.log_path

    for p in product_dict(**params):
        CONSTANT.cpv = p['cpv']
        CONSTANT.margin = p['margin'].item()
        CONSTANT.weight_decay = p['weight_decay']
        CONSTANT.embed_size = p['embed_size'].item()
        CONSTANT.learning_rate = p['learning_rate'].item()
        
        log_path =  os.path.join(o_log_path,
                    '{}cpv{}m{:.1f}wc{}wd100es{}lr{}'.format(CONSTANT.model, 
                    CONSTANT.cpv, CONSTANT.margin, CONSTANT.weight_decay, 
                    CONSTANT.embed_size, CONSTANT.learning_rate))

        # if the config is run, ignore it
        if os.path.isdir(log_path):
            continue

        CONSTANT.log_path = log_path

        CONSTANT.cap_train_path = 'cpv{}_jmet_glove/train.npy'.format(CONSTANT.cpv)
        CONSTANT.cap_val_path = 'cpv{}_jmet_glove/val.npy'.format(CONSTANT.cpv)
        CONSTANT.cap_test_path = 'cpv{}_jmet_glove/test.npy'.format(CONSTANT.cpv)

        train_vtt.main()

if __name__ == "__main__":
    main()