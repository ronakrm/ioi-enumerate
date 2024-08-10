import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

myBlue = '#005baa'
myRed = '#ff2a2a'
myPurple = '#25a000'
myGreen = '#ed00d4'
myYellow = '#f0e442'
myColors = [myBlue, myRed, myPurple, myGreen, myYellow]


def plot_hists(
    mydf: pd.DataFrame,
    col: str,
    val: str,
    plot_col: str = 'logit_diff',
    label_prefix="",
):

    group_1 = mydf[plot_col][mydf[col] == val]
    group_2 = mydf[plot_col][mydf[col] != val]

    # randomly select from group 2 to make the same size
    group_2 = group_2.sample(n=len(group_1), random_state=0)

    # Setup
    fig = plt.figure(figsize=(9,3), dpi=300, frameon=False)
    plt.tight_layout(pad=0.0)
    ax = fig.add_axes([0, 0, 1, 1])

    # clear everything except plot area
    # ax.axis('off')
    for item in [fig, ax]:
        item.patch.set_visible(False)
        item.patch.set_linewidth(0.0)
    ax.yaxis.set_visible(False)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.minorticks_off()

    plt.tick_params(
        which='both',      # both major and minor ticks are affected
        right=False,
        left=False,        # ticks along the bottom edge are off
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off

    sns.histplot(
        group_2,
        color=myPurple,
        bins=100,
        alpha=0.8,
        ax=ax,
    )

    sns.histplot(
        group_1,
        color=myBlue,
        bins=100,
        alpha=0.8,
        ax=ax,
    )

    plt.legend([f'{label_prefix}{val}', f'{label_prefix}not {val}'], loc='upper right')

    plt.savefig('figs/thing.png', bbox_inches='tight', pad_inches=0.0)


print('Loading data...')
df = pd.read_csv('small_logits.csv')
# Small
# # examples with logit diff < 0:	 125583 out of 9313920
# % examples with logit diff < 0:	 1.348%


print('Plotting...')
plot_hists(df, 'S', 'Lisa', plot_col='logit_diff', label_prefix="")

print('Plotting...')
s_df = df[df['S'] == 'Lisa']
plot_hists(s_df, 'IO', 'Alicia', plot_col='logit_diff', label_prefix="S: Lisa, ")

print('Loading data...')
df = pd.read_csv('medium_logits.csv')
# # examples with logit diff < 0:	 125583 out of 9313920
# % examples with logit diff < 0:	 1.348%



print('Plotting...')
plot_hists(df, 'S', 'Lisa', plot_col='logit_diff', label_prefix="")

print('Plotting...')
s_df = df[df['S'] == 'Lisa']
plot_hists(s_df, 'IO', 'Alicia', plot_col='logit_diff', label_prefix="S: Lisa, ")

print('Loading data...')
df = pd.read_csv('large_logits.csv')

print('Plotting...')
plot_hists(df, 'S', 'Lisa', plot_col='logit_diff', label_prefix="")

print('Plotting...')
s_df = df[df['S'] == 'Lisa']
plot_hists(s_df, 'IO', 'Alicia', plot_col='logit_diff', label_prefix="S: Lisa, ")

