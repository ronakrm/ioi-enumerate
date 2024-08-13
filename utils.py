import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Let's plot some of these extremes against the rest
def plot_against_alt(df, col, val, label_prefix=""):
    _, bins, _ = plt.hist(df['logit_diff'][df[col] != val], bins=100,  density=True, alpha=0.5, label=label_prefix+"All Others")
    plt.hist(df['logit_diff'][df[col] == val], bins=bins, density=True, alpha=0.5, label=f"{label_prefix}{col}:{val}")
    plt.legend()
    plt.show()


def get_first_order_df(df):
    subset_res_list = []
    # check all template subsets
    for col in ["template", "PLACE", "OBJECT", "S", "IO"]:
        print("Starting ", col)
        unique_vals = df[col].unique().tolist()
        for val in tqdm(unique_vals):
            subset_dict = {}
            subset_dict["col"] = col
            subset_dict["val"] = val
        
            subset_dict["mean"] = df['logit_diff'][df[col] == val].mean()
            subset_dict["std"] = df['logit_diff'][df[col] == val].std()
            subset_dict["alt_mean"] = df['logit_diff'][df[col] != val].mean()
            subset_dict["alt_std"] = df['logit_diff'][df[col] != val].std()
        
            subset_res_list.append(subset_dict)
    return pd.DataFrame(subset_res_list)


def compute_first_order_summary_stats(first_order_df, global_mean, global_std):    
    first_order_df["global_mean_diff"] = (first_order_df["mean"] - global_mean).abs()
    first_order_df["global_std_diff"] = (first_order_df["std"] - global_std).abs()
    first_order_df["relative_mean_diff"] = (first_order_df["mean"] - first_order_df["alt_mean"]).abs()
    first_order_df["relative_std_diff"] = (first_order_df["std"] - first_order_df["alt_std"]).abs()
    return first_order_df
    

def print_mean_and_std(df):
    # global mean and std
    global_mean = df['logit_diff'].mean()
    global_std = df['logit_diff'].std()
    
    print("Global mean:", global_mean)
    print("Global std:", global_std)

    return global_mean, global_std


def print_num_incorrect(df):
    # Count number with logit diff < 0
    print("# examples with logit diff < 0:\t", (df['logit_diff'] < 0).sum(), "out of", len(df))
    print("% examples with logit diff < 0:\t", f"{((df['logit_diff'] < 0).sum() / len(df))*100:.3f}%")