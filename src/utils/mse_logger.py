import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def update_mse_log(exp_name, model_name, train_mse, test_mse, output_dir):
    log_path = f"{output_dir}/mse_log.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new_row = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "exp_name": exp_name,
        "train_mse": train_mse,
        "test_mse": test_mse
    }
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(log_path, index=False)
    

model_colors = {
    "ridge": "#34A6F4",
    "xgboost": "#155DFC",
    "random_forest": "#7F22FE",
    "tabpfn": "#C6185C",
    "tabstar": "#FF8904"
}

def _sort_by_staretgy(df):
    def strategy_sort_key(series: pd.Series) -> pd.Series:
        return series.map(lambda s: (
            5 if "ensemble" in s else
            4 if "include_A_B" in s else
            4 if "raw_A_B" in s else
            3 if "A_B_only" in s else
            1.1 if "tfidf" in s else
            0 if "length" in s else
            2.1 if "llm" in s else
            2
        ))

    df = df.sort_values(
        by=["strategy", "exp_label"],
        key=lambda col: strategy_sort_key(col) if col.name == "strategy" else col)
    
    return df

def plot_mse(log_path):
    # Read the CSV
    all_data = pd.read_csv(log_path)

    # Create unique experiment label: (exp_name + datetime)
    all_data["exp_label"] = all_data["exp_name"] + " | " + all_data["datetime"]
    
    # Convert datetime column to actual datetime format (if not already)
    all_data["datetime"] = pd.to_datetime(all_data["datetime"])

    # Sort by datetime so newest is last
    all_data = all_data.sort_values("datetime")

    # Drop duplicates, keeping the newest (last one)
    all_data = all_data.drop_duplicates(subset="exp_name", keep="last")

    # Create strategy col which contains exp without model name
    all_data["strategy"] = all_data["exp_name"].str.replace(r"(ridge|xgboost|random_forest|tabpfn|tabstar)_", "", regex=True)

    # Melt to long format
    df_melted = all_data.melt(
        id_vars=["exp_label", "model_name", "strategy"],
        value_vars=["train_mse", "test_mse"],
        var_name="split",
        value_name="mse"
    )

    df_melted = _sort_by_staretgy(df_melted)

    # Setup subplots
    splits = ["train_mse", "test_mse"]
    fig, axs = plt.subplots(1, 2, figsize=(24, 12), sharey=True)

    for i, split in enumerate(splits):
        ax = axs[i]
        split_data = df_melted[df_melted["split"] == split]
        
        # Find minimum MSE value for this split
        min_mse = split_data["mse"].min()
        colors = [model_colors.get(model, f"C{i}") for model in split_data["model_name"]]
        sizes = [100 if mse == min_mse else 70 for mse in split_data["mse"]] # Highlight the lowest-MSE point(s)
        linewidths = [2 if mse == min_mse else 1 for mse in split_data["mse"]]
        
        ax.scatter(
            split_data["strategy"],
            split_data["mse"],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors='black',
            linewidths=linewidths
        )

        ax.set_title(split.replace("_mse", "").capitalize())
        ax.set_xlabel("Experiment")
        ax.set_ylim(0, 0.11)
        ax.axhline(y=0.05, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=0.04, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=0.03, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=0.02, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=0.01, color='darkgreen', linestyle='--', linewidth=1)
        ax.tick_params(axis='x', rotation=90)
        # add text above the y-axis line
        ax.text(0, 0.05, "0.05 (69%)", color='grey', fontsize=10, verticalalignment='bottom')
        ax.text(0, 0.04, "0.04 (75)", color='grey', fontsize=10, verticalalignment='bottom')
        ax.text(0, 0.03, "0.03 (83%)", color='grey', fontsize=10, verticalalignment='bottom')
        ax.text(0, 0.02, "0.02 (90%)", color='grey', fontsize=10, verticalalignment='bottom')
        ax.text(0, 0.01, "0.01 (100%)", color='darkgreen', fontsize=10, verticalalignment='bottom')

    # Manually create legend handles
    legend_handles = [
        mpatches.Patch(color=model_colors.get(model, f"C0"), label=model)
        for model in model_colors.keys()
    ]
    
    axs[0].set_ylabel("MSE")
    axs[1].legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(log_path.replace(".csv", ".png"))
    plt.close()
    
def plot_mse_by_model_family(log_path: str):
    all_data = pd.read_csv(log_path)

    # Normalize and prep
    all_data["datetime"] = pd.to_datetime(all_data["datetime"])
    all_data = all_data.sort_values("datetime").drop_duplicates(subset="exp_name", keep="last")
    all_data["exp_label"] = all_data["exp_name"] + " | " + all_data["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    all_data["strategy"] = all_data["exp_name"].str.replace(
        r"(ridge|xgboost|random_forest|tabpfn|tabstar)_", "", regex=True
    )

    # Long format
    df_melted = all_data.melt(
        id_vars=["exp_label","model_name","strategy","exp_name"],
        value_vars=["train_mse","test_mse"],
        var_name="split",
        value_name="mse"
    )
    # keep only test
    df_melted = df_melted[df_melted["split"] == "test_mse"].copy()

    # Precompute masks
    name_series = df_melted["exp_name"].astype(str)
    mask_distil = name_series.str.contains("distilroberta", case=False, na=False)
    mask_mpnet  = name_series.str.contains("mpnet", case=False, na=False)
    mask_minilm = name_series.str.contains("minilm", case=False, na=False)

    mask_any_family = mask_distil | mask_mpnet | mask_minilm
    # Rows with NO family token (e.g., "A_B_only", baselines)
    df_baseline = df_melted[~mask_any_family]

    family_map = {
        "all-distilroberta-v1": mask_distil,
        "all-mpnet-base-v2": mask_mpnet,
        "all-MiniLM-L6-v2": mask_minilm,
    }

    for fam, fam_mask in family_map.items():
        df_fam = pd.concat([df_melted[fam_mask], df_baseline], ignore_index=True)
        if df_fam.empty:
            continue

        df_fam = _sort_by_staretgy(df_fam)

        # find min for marker emphasis
        min_mse = df_fam["mse"].min()
        colors = [model_colors.get(m, "C0") for m in df_fam["model_name"]]
        sizes = [110 if v == min_mse else 70 for v in df_fam["mse"]]
        lw = [2 if v == min_mse else 1 for v in df_fam["mse"]]
        
        # remove the family name from the strategy for plotting
        df_fam["strategy"] = df_fam["strategy"].str.split(" | ").str[0]

        plt.figure(figsize=(9, 12))
        plt.scatter(
            df_fam["strategy"], df_fam["mse"],
            c=colors, s=sizes, alpha=0.7, edgecolors="black", linewidths=lw
        )

        # refs
        for val, txt, col in [
            (0.05, "0.05 (69%)", 'grey'),
            (0.04, "0.04 (75%)", 'grey'),
            (0.03, "0.03 (83%)", 'grey'),
            (0.02, "0.02 (90%)", 'grey'),
            (0.01, "0.01 (100%)", 'darkgreen'),
        ]:
            plt.axhline(y=val, color=col, linestyle='--', linewidth=1)
            plt.text(0, val, txt, color=col, fontsize=9, va='bottom')
        
        plt.title(f"Test MSE: {fam}")
        plt.xlabel("Experiment")
        plt.ylabel("MSE")
        plt.ylim(0, 0.11)
        plt.xticks(rotation=90)
        
        # Manually create legend handles
        legend_handles = [
            mpatches.Patch(color=model_colors.get(model, f"C0"), label=model)
            for model in model_colors.keys()
        ]
    
        plt.legend(handles=legend_handles, loc="upper right")

        plt.tight_layout()
        out_png = log_path.replace(".csv", f"_{fam}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
    
if __name__ == "__main__":
    # Example usage
    plot_mse("src/results/mse_log.csv")
    plot_mse_by_model_family("src/results/mse_log.csv")