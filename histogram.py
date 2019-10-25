import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(df, feature):
    plt.figure(figsize=(8, 6))
    df.groupby('Hogwarts House')[feature].plot(kind='hist', alpha=.5);
    plt.legend();
    plt.title(feature)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to consider')
    parser.add_argument('--all', '-a', default=False, action='store_true', help="display all histograms")

    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset, index_col=0)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    if 'Hogwarts House' not in df.columns:
        print("No target in dataset")
        exit(1)
    
    if args.all:
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
        courses = ['Arithmancy', 'Care of Magical Creatures']
    
    for course in courses:
        plot_histogram(df, course)
    print("Homogeneous scores: Arithmancy, Care of Magical Creatures")
