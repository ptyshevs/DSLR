import pandas as pd
import numpy as np
import pickle
import argparse
from common import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='accept dataset for training')
    parser.add_argument('--lr', '-l', help='learning rate', default=.05)
    parser.add_argument('--normalize', '-n', help='normalize dataset before training', default=False, action='store_true')
    parser.add_argument('--all-features', '-a', help='use all features instead of the selected ones', default=False, action='store_true')
    parser.add_argument('--seed', '-s', help='random seed', default=0)
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='verbose mode')
    parser.add_argument('--epochs', '-e', help='# of epochs', default=3)
    parser.add_argument('--batch-size', '-b', help='Batch size [1 for SGD, <n for mini-batch GD]', default=32)
    parser.add_argument('--save-path', '-p', default='weights.pcl', help="File name to save weights")

    args = parser.parse_args()
    if type(args.lr) is str:
        args.lr = float(args.lr)
    if type(args.seed) is str:
        args.seed = int(args.seed)
    if type(args.epochs) is str:
        args.epochs = int(args.epochs)
    if type(args.batch_size) is str:
        args.batch_size = int(args.batch_size)

    try:
        df = pd.read_csv(args.dataset, index_col=0)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    target = 'Hogwarts House'

    if args.all_features:
        courses = ['Arithmancy',
                    'Astronomy',
                    'Herbology',
                    'Defense Against the Dark Arts',
                    'Divination',
                    'Muggle Studies',
                    'Ancient Runes',
                    'History of Magic',
                    'Transfiguration',
                    'Potions',
                    'Care of Magical Creatures',
                    'Charms',
                    'Flying']
    else:
        courses = ['Herbology', 'Defense Against the Dark Arts',
                   'Ancient Runes', 'Charms']

    X = df[courses].values
    y = df[target]
    
    X, imp_vec = impute(X)
    
    if args.normalize:
        Xv, X_mean, X_std = normalize(X)
    else:
        Xv, X_mean, X_std = X, X.mean(axis=0), X.std(axis=0)
    
    lr = LogisticRegression(verbose=args.verbose, seed=args.seed)
    lr.fit(Xv, y, batch_size=args.batch_size, n_epochs=args.epochs)
    
    store_obj = {'model': lr, 'imp': imp_vec, 'X_mean': X_mean, 'X_std': X_std}
    with open(args.save_path, 'wb') as f:
        print(f"Model is saved in {args.save_path}")
        pickle.dump(store_obj, f)