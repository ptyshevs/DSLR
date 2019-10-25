import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_scatter(df, feature1, feature2):
    plt.figure(figsize=(8, 6))
    if 'Hogwarts House' in df.columns:
        c = df['Hogwarts House'].map({'Ravenclaw': 'b', 'Slytherin': 'g', 'Gryffindor': 'r', 'Hufflepuff': 'orange'})
    else:
        c = None
    plt.scatter(df[feature1], df[feature2], c=c, alpha=.4)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'{feature1} vs {feature2}')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to consider')
    parser.add_argument('--all', '-a', default=False, action='store_true', help="display all scatters")

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
        courses = ['Astronomy', 'Defense Against the Dark Arts']
    
    n = len(courses)
    for xi in range(n):
        for yj in range(xi, n):
            x = courses[xi]
            y = courses[yj]
            if x != y:
                plot_scatter(df, x, y)
    print("Similar features: Astronomy and Defence Against the Dark Arts")
