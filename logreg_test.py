import numpy as np
import pandas as pd
from common import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to predict for')
    parser.add_arugment('--filename', '-f', default='predictions.csv', help='file to save predictions for')
    parser.add_argument('--all-features', '-a', help='use all features instead of the selected ones', default=False, action='store_true')

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
