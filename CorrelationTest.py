from src import LoadDatav2 as ld
import matplotlib.pyplot as plt
import seaborn as sns

'''
Function plot_corr a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
'''


def plot_corr(corr, size=10):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(df)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.show()


def plot_corr_seaborn(corr):
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.tight_layout()
    plt.show()


# Get diagonal and lower triangular pairs of correlation matrix
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


# Get ordered correlations
def get_top_abs_correlations(df, n=5):
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# Retrieves data
data = ld.retdata()
# Pearsons correlation test
corrmatrix = data.corr(method='pearson')

# Plot using matplotlib or seaborn package
# plot_corr(corrmatrix)
plot_corr_seaborn(corrmatrix)

# Print correlation matrix, and order correlations with top_abs_correlations
print(corrmatrix)
print()
print(get_top_abs_correlations(corrmatrix, 10))
