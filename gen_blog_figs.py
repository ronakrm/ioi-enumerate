import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

myBlue = '#005baa'
myRed = '#ff2a2a'
myPurple = '#25a000'
myGreen = '#ed00d4'
myYellow = '#f0e442'
myColors = [myBlue, myRed, myPurple, myGreen, myYellow]


def plot_full_hist(
    mydf: pd.DataFrame,
    plot_col: str = 'logit_diff',
    plot_label="",
    outname="figs/test.png",
    n=10000,
):
    sampled_df = mydf[plot_col].sample(n=n, random_state=0)

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
        sampled_df,
        color=myPurple,
        bins=100,
        alpha=0.8,
        ax=ax,
    )

    plt.legend([f"{plot_label} (n={n})"], loc='upper right')

    plt.savefig(outname, bbox_inches='tight', pad_inches=0.0)



def plot_hists(
    mydf: pd.DataFrame,
    col: str,
    val: str,
    plot_col: str = 'logit_diff',
    label_prefix="",
    outname="figs/test.png",
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

    plt.savefig(outname, bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':

    print('Loading data...')
    df = pd.read_csv('results/small_logits.csv')
    # Small
    # # examples with logit diff < 0:	 125583 out of 9313920
    # % examples with logit diff < 0:	 1.348%
    # S: Lisa, then IO: Alicia
    
    print('Plotting...')
    plot_full_hist(df, n=100000, plot_label="GPT-2 Small Full BABA IOI Logit Diffs", outname='figs/small_full.png')
    
    print('Plotting...')
    plot_hists(df, 'S', 'Lisa', plot_col='logit_diff', label_prefix="", outname='figs/small_s.png')
    
    print('Plotting...')
    s_df = df[df['S'] == 'Lisa']
    plot_hists(s_df, 'IO', 'Alicia', plot_col='logit_diff', label_prefix="S: Lisa, ", outname='figs/small_s_io.png')

    # Medium
    # # examples with logit diff < 0:	 4143 out of 9313920
    # % examples with logit diff < 0:	 0.044%

    # print('Loading data...')
    # df = pd.read_csv('results/medium_logits.csv')
    # 
    # print('Plotting...')
    # plot_full_hist(df, n=100000, plot_label="GPT-2 Medium Full BABA IOI Logit Diffs", outname='figs/med_full.png')

    # Large
    # # examples with logit diff < 0:	 5986 out of 9313920
    # % examples with logit diff < 0:	 0.064%

    # print('Loading data...')
    # df = pd.read_csv('results/large_logits.csv')

    # print('Plotting...')
    # plot_full_hist(df, n=100000, plot_label="GPT-2 Large Full BABA IOI Logit Diffs", outname='figs/large_full.png')
