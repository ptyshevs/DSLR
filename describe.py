import argparse
import pandas as pd

import math

def count(f):
    n = 0
    for c in f:
        n += not math.isnan(c)
    return n

def mean(f):
    f = list(filter(lambda x: not math.isnan(x), f))
    if len(f) > 0:
        return sum(f) / count(f)
    return float('nan')

def std(f):
    f_mean = mean(f)
    return mean([(c - f_mean) ** 2 for c in f]) ** .5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to consider')

    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset, index_col=0)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    columns = []
    statistics = {'count':[],
                'mean': [],
                'std': [],
                'min': [],
                '25%': [],
                '50%': [],
                '75%': [],
                'max': []}
    for c, d in zip(df.dtypes.index, df.dtypes):
        d = str(d)
        if d.startswith('int') or d.startswith("float"):
            columns.append(c)
            
            feature = df[c]
            statistics['count'].append(count(feature))
            feature = [c for c in feature if not math.isnan(c)]
            feature = sorted(feature)
            n = len(feature)
            
            if n > 0:
                f_min = feature[0]
                f_max = feature[-1]
                f_25 = feature[int(n * .25)]
                f_50 = feature[int(n * .5)]
                f_75 = feature[int(n * .75)]
            else:
                f_min = f_max = f_25 = f_50 = f_75 = float('nan')
            
            statistics['mean'].append(mean(feature))
            statistics['std'].append(std(feature))
            statistics['min'].append(f_min)
            statistics['25%'].append(f_25)
            statistics['50%'].append(f_50)
            statistics['75%'].append(f_75)
            statistics['max'].append(f_max)

    print(pd.DataFrame(statistics, index=columns).T)