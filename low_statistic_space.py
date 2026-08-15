import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, accuracy_score
import config
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent # Path to folder opened files

_CS_MAP = {
    "LAB": (cv2.COLOR_BGR2LAB, {"L": 0, "A": 1, "B": 2}),
    "HSV": (cv2.COLOR_BGR2HSV, {"H": 0, "S": 1, "V": 2}),
    "YCRCB": (cv2.COLOR_BGR2YCrCb, {"Y": 0, "CR": 1, "CB": 2}),
}

STAT_METRICS = ["mean", "skew", "kurtosis", "entropy"]
PERF_METRICS = ["Accuracy", "F1"]
MODELS = ["ResNet50", "MobileNetV2", "ViT16"]
TARGETS = [
    ("LAB", "L"),
    ("HSV", "V"),
    ("YCRCB", "Y"),
]


def distribution_stats(values: np.ndarray) -> dict:

    values = np.asarray(values).ravel()
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(float)
    pdf = hist / hist.sum() if hist.sum() else hist

    return {"mean": values.mean(), "skew": stats.skew(values), "kurtosis": stats.kurtosis(values), "entropy": stats.entropy(pdf),}


def iter_channel_arrays(dir_path: Path, color_space: str, channel: str,):
    conversion, channel_map = _CS_MAP[color_space]
    channel_idx = channel_map[channel]

    for image_path in Path(dir_path).iterdir():
        image = cv2.imread(str(image_path))
        converted = cv2.cvtColor(image, conversion)
        yield converted[:, :, channel_idx]

def calculate_class_channel_stats(dir_path: Path, color_space: str, channel: str,) -> dict:

    channel_arrays = list(iter_channel_arrays(dir_path, color_space, channel))
    pixels = np.concatenate([img.ravel() for img in channel_arrays])
    return distribution_stats(pixels)

def build_color_stats_df(class_names: list[str], targets: list[tuple[str, str]],) -> pd.DataFrame:

    rows = []

    db_dirs = {"REAL": "cifake1/train/REAL", "FAKE1": "cifake1/train/FAKE", "FAKE2": "cifake2/train/FAKE",}

    for db_name, db_path in db_dirs.items():
        for class_name in class_names:
            class_dir = PROJECT_ROOT / f"{db_path}/{class_name}/"

            for color_space, channel in targets:
                stat_row = calculate_class_channel_stats(class_dir, color_space, channel,)
                rows.append({"DB": db_name, "class_name": class_name, "color_space": color_space, "ch": channel, **stat_row,})

    return pd.DataFrame(rows)


def load_performance_df(seeds) -> pd.DataFrame:

    rows = []

    for db_name in ['REAL', 'FAKE1', 'FAKE2',]:
        for model in MODELS:
            for seed in seeds:
                csv_path = PROJECT_ROOT / "results" / f"{model}_{db_name}_REAL_{seed}.csv"
                result_df = pd.read_csv(csv_path)

                y_true = result_df["label"]
                y_pred = result_df["preds"]

                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )

                for label_text, scores in report.items():
                    if not label_text.isdigit():
                        continue
                    label = int(label_text)

                    mask = (y_true == label)

                    acc = accuracy_score(
                        y_true[mask],
                        y_pred[mask]
                    )

                    rows.append({
                        "DB": db_name, "model": model, "label": label, "class_name": config.label_to_class[label], "Accuracy": acc, "F1": scores["f1-score"],})

    performance_df = pd.DataFrame(rows)
    return (performance_df.groupby(["DB", "model", "label", "class_name"], as_index=False,)[PERF_METRICS].mean())

def build_performance_diff_df(performance_df: pd.DataFrame, comparison_db: str, reference_db: str = "REAL",) -> pd.DataFrame:

    index_cols = ["model", "label", "class_name"]
    reference = (performance_df.query("DB == @reference_db").set_index(index_cols)[PERF_METRICS])
    comparison = (performance_df.query("DB == @comparison_db").set_index(index_cols)[PERF_METRICS])
    diff = reference.subtract(comparison).abs()
    diff.columns = [f"{metric}_diff" for metric in diff.columns]
    return (diff.reset_index().assign(DB1=reference_db, DB2=comparison_db))

def build_analysis_df(color_stats_df: pd.DataFrame, performance_diff_df: pd.DataFrame, comparison_db: str, reference_db: str = "REAL",) -> pd.DataFrame:

    long_stats = color_stats_df.melt(
        id_vars=["DB", "class_name", "color_space", "ch"],
        value_vars=STAT_METRICS,
        var_name="stat_metric",
        value_name="stat_value",
    )

    stat_wide = long_stats.pivot_table(index=["class_name", "color_space", "ch", "stat_metric"], columns="DB", values="stat_value", aggfunc="mean",)
    color_diff_df = (stat_wide[reference_db].subtract(stat_wide[comparison_db]).abs().rename("color_diff").reset_index())
    return performance_diff_df.merge(color_diff_df, on="class_name", how="left",)

def safe_correlations(x: np.ndarray, y: np.ndarray,) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan,}

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    return {"n": n, "pearson_r": pearson_r, "pearson_p": pearson_p, "spearman_r": spearman_r, "spearman_p": spearman_p,}


def make_corr_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["color_space", "ch", "stat_metric", "DB1", "DB2", "model",]

    rows = []

    for keys, group in analysis_df.groupby(group_cols):
        metadata = dict(zip(group_cols, keys))
        x = group["color_diff"].to_numpy()

        for metric in PERF_METRICS:
            correlation = safe_correlations(x, group[f"{metric}_diff"].to_numpy(),)
            rows.append({**metadata, "metric": metric, **correlation,})

    return (pd.DataFrame(rows).sort_values(group_cols + ["metric"]).reset_index(drop=True))

def plot_performance_gap(
    analysis_df: pd.DataFrame,
    metric: str = "F1",
    annot_model: str | None = "MobileNetV2",
):
    stat_metrics = analysis_df["stat_metric"].unique()
    models = analysis_df["model"].unique()

    ncols = 4
    nrows = int(np.ceil(len(stat_metrics) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    DB_COLORS = {
        "FAKE1": "tab:blue",  # SDGen
        "FAKE2": "tab:orange",  # EDMGen
    }

    model_markers = {
        "MobileNetV2": "o",
        "ResNet50": "^",
        "ViT16": "s",
    }

    db_label_map = {
        "FAKE1": "SDGen",
        "FAKE2": "EDMGen",
    }

    y_col = f"{metric}_diff"

    for i, stat_metric in enumerate(stat_metrics):
        ax = axes[i]

        stat_df = analysis_df.query(
            "stat_metric == @stat_metric"
        )

        # FAKE1 / FAKE2
        for db2, db_df in stat_df.groupby("DB2"):

            for model, model_df in db_df.groupby("model"):

                ax.scatter(
                    model_df["color_diff"],
                    model_df[y_col],
                    marker=model_markers[model],
                    color=DB_COLORS[db2],
                    linewidths=1.2,
                    s=70,
                    alpha=0.9,
                )

                # class annotation
                if model == annot_model:
                    for _, row in model_df.iterrows():
                        ax.annotate(
                            row["class_name"],
                            (row["color_diff"], row[y_col]),
                            xytext=(4, 3),
                            textcoords="offset points",
                            fontsize=12,
                        )

        ax.set_title(stat_metric,fontsize=18)
        ax.set_xlabel("Distribution difference",fontsize=14)
        if i % ncols == 0:
            ax.set_ylabel(y_col,fontsize=14)
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.25)

    for ax in axes[len(stat_metrics):]:
        ax.axis("off")

    # Model legend: marker shape
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[model],
            color="black",
            linestyle="",
            markersize=10,
            label=model,
        )
        for model in models
    ]

    # Dataset legend: color
    db_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DB_COLORS["FAKE1"],
            linestyle="",
            markersize=10,
            label="SDGen",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DB_COLORS["FAKE2"],
            linestyle="",
            markersize=10,
            label="EDMGen",
        ),
    ]

    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.38, 0.01),
        ncol=len(model_handles),
        frameon=False,
        title="Model",
        fontsize=14,
        title_fontsize=14
    )

    fig.legend(
        handles=db_handles,
        loc="lower center",
        bbox_to_anchor=(0.78, 0.01),
        ncol=2,
        frameon=False,
        title="Dataset",
        fontsize=14,
        title_fontsize=14
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.subplots_adjust(wspace=0.12)
    return fig

def make_paper_table(
    corr_df: pd.DataFrame,
    comparison_db: str,
    corr_type: str = "pearson",
) -> pd.DataFrame:

    r_col = f"{corr_type}_r"
    p_col = f"{corr_type}_p"

    model_order = ["MobileNetV2", "ResNet50", "ViT16",]
    metric_order = ["F1", "Accuracy",]
    stat_order = ["mean", "entropy", "kurtosis", "skew",]
    target_order = [("HSV", "V"), ("LAB", "L"), ("YCRCB", "Y"),]

    df = corr_df[
        (corr_df["DB1"] == "REAL") &
        (corr_df["DB2"] == comparison_db)
    ].copy()

    df = df[
        df.apply(
            lambda row: (row["color_space"], row["ch"]) in target_order,
            axis=1
        )
    ].copy()

    target_name = {("HSV", "V"): "HSV(V)", ("LAB", "L"): "CIELab(L)", ("YCRCB", "Y"): "YCrCb(Y)",}
    stat_name = {"mean": "Mean", "entropy": "Ent.", "kurtosis": "Kurt.", "skew": "Skew",}

    df["target"] = df.apply(
        lambda row: target_name[(row["color_space"], row["ch"])],
        axis=1
    )

    df["stat"] = df["stat_metric"].map(stat_name)
    df["model"] = pd.Categorical(
        df["model"],
        categories=model_order,
        ordered=True
    )

    df["metric"] = pd.Categorical(
        df["metric"],
        categories=metric_order,
        ordered=True
    )

    df["r_p"] = df.apply(
        lambda row: (
            f"{row[r_col]:.2f} (p={row[p_col]:.3f})"
            if pd.notna(row[r_col]) and pd.notna(row[p_col])
            else ""
        ),
        axis=1
    )

    table = df.pivot_table(
        index=["model", "metric"],
        columns=["target", "stat"],
        values="r_p",
        aggfunc="first",
        observed=True,
    )

    desired_columns = []

    for color_space, ch in target_order:
        target = target_name[(color_space, ch)]

        for stat in stat_order:
            desired_columns.append(
                (target, stat_name[stat])
            )

    table = table.reindex(columns=desired_columns)
    table = table.sort_index()
    table = table.round(2)

    return table

def main(seeds: list[int]) -> None:

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_corr = []
    all_analysis = []

    color_stats_df = build_color_stats_df(
        class_names=list(config.label_to_class.values()),
        targets=TARGETS,
    )

    performance_df = load_performance_df(seeds)

    DBS = ["FAKE1", "FAKE2"]

    for db in DBS:

        performance_diff_df = build_performance_diff_df(
            performance_df=performance_df,
            comparison_db=db,
        )

        analysis_df = build_analysis_df(
            color_stats_df=color_stats_df,
            performance_diff_df=performance_diff_df,
            comparison_db=db,
        )

        all_analysis.append(analysis_df)

        corr_df = make_corr_table(analysis_df)
        all_corr.append(corr_df)

    # =========================================
    # FAKE1 + FAKE2
    # =========================================
    analysis_all_df = pd.concat(
        all_analysis,
        ignore_index=True
    )

    # =========================================
    # Combined scatter plots
    # =========================================
    for color_space, channel in TARGETS:

        target_df = analysis_all_df.query(
            "color_space == @color_space and ch == @channel"
        )

        fig = plot_performance_gap(
            target_df,
            metric="F1",
        )

        plot_path = (
            output_dir
            / f"gap_{color_space}_{channel}_REAL_FAKE1_FAKE2_F1.png"
        )

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("saved:", plot_path)

    # =========================================
    # Correlation tables
    # =========================================
    corr_all_df = pd.concat(
        all_corr,
        ignore_index=True
    )

    corr_path = (
        output_dir
        / "corr_ALL_REAL_FAKE1_FAKE2.csv"
    )

    corr_all_df.to_csv(
        corr_path,
        index=False
    )

    print("saved:", corr_path)

    # SDGen
    sdgen_table = make_paper_table(
        corr_all_df,
        comparison_db="FAKE1",
        corr_type="pearson",
    )

    sdgen_path = (
        output_dir
        / "table_correlation_SDGen.csv"
    )

    sdgen_table.to_csv(sdgen_path)

    # EDMGen
    edmgen_table = make_paper_table(
        corr_all_df,
        comparison_db="FAKE2",
        corr_type="pearson",
    )

    edmgen_path = (
        output_dir
        / "table_correlation_EDMGen.csv"
    )

    edmgen_table.to_csv(edmgen_path)

if __name__ == "__main__":
    seeds = [12, 123, 1234,]
    main(seeds)