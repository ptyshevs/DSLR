import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def plot_pairplot(df, features, target=None, args=None):
    if target is None:
        g = sns.pairplot(df[features])
    else:
        g = sns.pairplot(df[features + [target]], hue=target, diag_kind='hist', plot_kws={'alpha': .5}, diag_kws={'alpha': .5})
    g.map_lower(hide_current_axis)
    xlabels,ylabels = [],[]

    for ax in g.axes[-1,:]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
    for ax in g.axes[:,0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)

    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            g.axes[j,i].xaxis.set_label_text(xlabels[i])
            g.axes[j,i].yaxis.set_label_text(ylabels[j])

    plt.savefig(args.filename)
    if not args.silent:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to consider')
    parser.add_argument('--filename', '-f', default='pair_plot.png', help='filename where to save plot')
    parser.add_argument('--silent', '-s', default=False, action='store_true', help='figure is saved only')

    args = parser.parse_args()
    try:
        df = pd.read_csv(args.dataset, index_col=0)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    if 'Hogwarts House' not in df.columns:
        print("No target in dataset")
        exit(1)
    
    target = 'Hogwarts House'
    features = [c for c in df.columns if c != target]
    
    plot_pairplot(df, features, target, args)
    print("I will use Herbology, Defense Against the Dark Arts, Ancient Runes, and Charms")