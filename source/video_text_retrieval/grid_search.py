
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
        'margin': np.linspace(0, 1, 6),
        'weight_decay': [0, *np.logspace(-8, -12, 3)],
        # 'word_dim': [50, 100, 200, 300],
        # 'word2vec_path': [""],
        'embed_size': np.linspace(256, 1536, 6, dtype=int),
        'learning_rate': np.logspace(-1, -5, 5, dtype=float)
    }

    o_log_path = CONSTANT.log_path

    for p in product_dict(**params):
        CONSTANT.cpv = p['cpv']
        CONSTANT.margin = p['margin'].item()
        CONSTANT.weight_decay = p['weight_decay']
        CONSTANT.embed_size = p['embed_size'].item()
        CONSTANT.learning_rate = p['learning_rate'].item()
        
        log_path =  os.path.join(o_log_path,
                    f'{CONSTANT.model}cpv{CONSTANT.cpv}m{CONSTANT.cpv}'\
                    f'wc{CONSTANT.weight_decay}wd100es{CONSTANT.embed_size}'\
                    f'lr{CONSTANT.learning_rate}')

        # if the config is run, ignore it
        if os.path.isdir(log_path):
            continue

        CONSTANT.log_path = log_path

        CONSTANT.cap_train_path = f'cpv{CONSTANT.cpv}_jmet_glove/train.npy'
        CONSTANT.cap_val_path = f'cpv{CONSTANT.cpv}_jmet_glove/val.npy'
        CONSTANT.cap_test_path = f'cpv{CONSTANT.cpv}_jmet_glove/test.npy'

        train_vtt.main()

if __name__ == "__main__":
    main()