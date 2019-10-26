import numpy as np
import pandas as pd
from common import *
import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to predict for')
    parser.add_argument('--weights', '-w', default='weights.pcl', help='file to load weights from')
    parser.add_argument('--filename', '-f', default='houses.csv', help='file to save predictions for')
    parser.add_argument('--all-features', '-a', help='use all features instead of the selected ones', default=False, action='store_true')
    parser.add_argument('--normalize', '-n', help='normalize dataset', default=False, action='store_true')
    parser.add_argument('--impute', '-i', help='impute NaN values with mean', default=False, action='store_true')
    args = parser.parse_args()

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

    df = pd.read_csv(args.dataset, index_col=0)

    with open(args.weights, 'rb') as f:
        store_obj = pickle.load(f)
    
    lr = store_obj['model']
    imp_vec = store_obj['imp']
    X_mean = store_obj['X_mean']
    X_std = store_obj['X_std']

    
    X = df[courses].values
    y = df[target]
    
    if args.impute:
        X, imp_vec = impute(X, vec=imp_vec)

    if args.normalize:
        X = (X - X_mean) / X_std
    y_pred = lr.predict(X)
    pred_df = pd.DataFrame(y_pred, index=df.index, columns=[target])
    pred_df.to_csv(args.filename)
    print("Predictions are saved in", args.filename)
