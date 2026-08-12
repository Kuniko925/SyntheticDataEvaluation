import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
from matplotlib.lines import Line2D
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt


def plot_improvement_ci_distribution(
    summary_df,
    from_ratio=20,
    to_ratio=30,
    save_path=None,
):

    target = summary_df[
        summary_df["mix_ratio"].isin(
            [from_ratio, to_ratio]
        )
    ].copy()

    wide = target.pivot(
        index=["model", "class_id"],
        columns="mix_ratio",
        values="f1_mean"
    ).dropna()

    improvements = (
        wide[to_ratio] - wide[from_ratio]
    ).to_numpy()

    n = len(improvements)

    mean_diff = improvements.mean()
    std_diff = improvements.std(ddof=1)

    sem = std_diff / np.sqrt(n)

    # 95% CI
    tcrit = t.ppf(
        0.975,
        df=n - 1
    )

    ci_low = mean_diff - tcrit * sem
    ci_high = mean_diff + tcrit * sem

    x = np.linspace(
        mean_diff - 4 * sem,
        mean_diff + 4 * sem,
        1000
    )

    z = (x - mean_diff) / sem

    y = t.pdf(
        z,
        df=n - 1
    ) / sem

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        x,
        y,
        color="blue",
        linewidth=1.8
    )

    # 95% CI
    mask = (
        (x >= ci_low) &
        (x <= ci_high)
    )

    ax.fill_between(
        x[mask],
        0,
        y[mask],
        alpha=0.30,
        label="95% CI"
    )

    # mean
    ax.axvline(
        mean_diff,
        color="red",
        linestyle="-",
        linewidth=1,
        label=f"Mean = {mean_diff:.3f}"
    )

    # CI boundary
    ax.axvline(
        ci_low,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"CI Lower = {ci_low:.3f}"
    )

    ax.axvline(
        ci_high,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"CI Upper = {ci_high:.3f}"
    )

    ax.set_xlabel(
        f"F1 Improvement ({from_ratio}% → {to_ratio}%)",
        fontsize=13
    )

    ax.set_ylabel(
        "Density",
        fontsize=13
    )

    ax.legend(
        frameon=False,
        loc="upper right"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print("saved:", save_path)

    plt.close(fig)

    print(
        f"{from_ratio}% -> {to_ratio}% improvement:"
    )
    print(f"n = {n}")
    print(f"mean = {mean_diff:.4f}")
    print(
        f"95% CI = "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )
def main(models, ratios, seeds, dir):

    rows = []

    for model in models:
        for r in ratios:
            for seed in seeds:

                if r == 0:
                    csv_path = dir / f"{model}_FAKE1_REAL_{seed}.csv"
                elif r == 100:
                    csv_path = dir / f"{model}_REAL_REAL_{seed}.csv"
                else:
                    csv_path = dir / f"{model}_MIX_{r}_{seed}.csv"

                df = pd.read_csv(csv_path)

                rep = classification_report(
                    df["label"],
                    df["preds"],
                    output_dict=True,
                    zero_division=0
                )

                for k, v in rep.items():

                    if not str(k).isdigit():
                        continue

                    rows.append({
                        "model": model,
                        "mix_ratio": r,
                        "seed": seed,
                        "class_id": int(k),
                        "f1": v["f1-score"],
                    })

    report_df = pd.DataFrame(rows)

    summary_df = (
        report_df
        .groupby(
            ["model", "class_id", "mix_ratio"],
            as_index=False
        )["f1"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "f1_mean",
                "std": "f1_std",
            }
        )
    )

    summary_df = summary_df.sort_values(
        ["model", "class_id", "mix_ratio"]
    ).copy()

    summary_df["delta_f1"] = (
        summary_df
        .groupby(["model", "class_id"])["f1_mean"]
        .diff()
    )

    plot_improvement_ci_distribution(
        summary_df,
        from_ratio=20,
        to_ratio=30,
        save_path=(
                dir /
                "f1_improvement_20_30_ci95.png"
        )
    )

    plot_improvement_ci_distribution(
        summary_df,
        from_ratio=40,
        to_ratio=50,
        save_path=(
                dir /
                "f1_improvement_40_50_ci95.png"
        )
    )

    plot_improvement_ci_distribution(
        summary_df,
        from_ratio=60,
        to_ratio=70,
        save_path=(
                dir /
                "f1_improvement_60_70_ci95.png"
        )
    )

    MODEL_COLORS = {
        "ResNet50": "tab:blue",
        "MobileNetV2": "tab:orange",
        "ViT16": "tab:green",
    }

    classes = list(range(10))
    ratios = sorted(summary_df["mix_ratio"].unique())

    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 7),
        sharex=True,
        sharey=True
    )

    for idx, c in enumerate(classes):

        ax = axes[idx // 5, idx % 5]

        ax.set_title(
            config.label_to_class[c],
            fontsize=14
        )

        ax.set_ylim(0, 1)

        for model in models:
            dm = summary_df[
                (summary_df["model"] == model) &
                (summary_df["class_id"] == c)
                ].copy()

            dm = dm.sort_values("mix_ratio")

            x = dm["mix_ratio"].to_numpy()
            mean = dm["f1_mean"].to_numpy()
            std = dm["f1_std"].to_numpy()

            ax.plot(
                x,
                mean,
                color=MODEL_COLORS[model],
                marker="o",
                linewidth=1.8,
                markersize=5,
            )

            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=MODEL_COLORS[model],
                alpha=0.15,
            )

        ax.axvline(
            x=30,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.25,
            zorder=0,
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        if idx // 5 == 1:
            ax.set_xlabel(
                "REAL Ratio (%)",
                fontsize=13
            )

        if idx % 5 == 0:
            ax.set_ylabel(
                "F1",
                fontsize=13
            )



    legend_handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=model,
        )
        for model in models
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0.07, 1, 1)
    )

    save_path = dir / "all_mix_f1_mean_std_lines.png"

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("saved:", save_path)

if __name__== "__main__":

    models = ["ResNet50", "MobileNetV2", "ViT16"]
    ratios = [0] + list(range(10, 100, 10)) + [100]
    seeds = [12, 123, 1234]
    dir = Path(config.PROJECT_ROOT) / "results"
    main(models, ratios, seeds, dir)



