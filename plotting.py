from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import basedata
from scipy.stats import pearsonr, spearmanr

def plot_confusion_matrix_from_performance(
    csv_path, label_to_class, title="", normalize=None, ax=None, add_colorbar=False):
    df = pd.read_csv(csv_path)
    y_true = df["label"].to_numpy()
    y_pred = df["preds"].to_numpy()

    labels = sorted(label_to_class.keys())
    class_names = [label_to_class[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    if add_colorbar:
        fig.colorbar(im, ax=ax)

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = np.nanmax(cm) * 0.6 if np.size(cm) else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    if created_fig:
        fig.tight_layout()

    return cm, class_names, im


def make_gap_figure(
        DB1: str,
        DB2_list: list[str],
        color_stats_df,
        ch: str,
        stat_metric: str,
):
    dfs = []
    for DB2 in DB2_list:
        color_diff_df = basedata.get_statistical_color_diff(
            color_stats_df,
            stat_metric=stat_metric,
            db_a=DB1,
            db_b=DB2
        )

        perf_diff_df = basedata.get_performance_diff(DB1, DB2)
        df = perf_diff_df.merge(color_diff_df, on="class_name", how="left")
        df["DB2"] = DB2
        dfs.append(df)

    plot_df = pd.concat(dfs, ignore_index=True)

    required_cols = {"class_name", "model", "color_diff", "Precision_diff", "Recall_diff", "F1_diff", "DB2"}
    missing = required_cols - set(plot_df.columns)
    if missing:
        raise ValueError(f"plot_df is missing required columns: {missing}")

    metrics = ["Precision_diff", "Recall_diff", "F1_diff"]
    metric_labels = {"Precision_diff": "Precision", "Recall_diff": "Recall", "F1_diff": "F1"}
    metric_colors = {"Precision_diff": "tab:blue", "Recall_diff": "tab:orange", "F1_diff": "tab:green"}

    plot_df = plot_df.dropna(subset=["color_diff"] + metrics)

    models_order = list(plot_df["model"].unique())
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    model_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(models_order)}

    x = plot_df["color_diff"].to_numpy()
    y_all = np.concatenate([plot_df[m].to_numpy() for m in metrics])

    xpad = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 1.0
    ypad = (y_all.max() - y_all.min()) * 0.05 if y_all.max() > y_all.min() else 0.01
    xlim = (x.min() - xpad, x.max() + xpad)
    ylim = (y_all.min() - ypad, y_all.max() + ypad)

    nrows = len(DB2_list)
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    for r, DB2 in enumerate(DB2_list):
        g_db = plot_df[plot_df["DB2"] == DB2]
        for c, met in enumerate(metrics):
            ax = axes[r, c]
            for model, g_m in g_db.groupby("model"):
                ax.scatter(
                    g_m["color_diff"], g_m[met],
                    color=metric_colors[met],
                    marker=model_markers[model],
                    s=55, alpha=0.85, linewidth=0.4
                )

            for _, row in g_m.iterrows():
                ax.annotate(
                    str(row["class_name"]),
                    (row["color_diff"], row[met]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, alpha=0.85
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)

            if r == 0:
                ax.set_title(metric_labels[met])
            if c == 0:
                ax.set_ylabel(f"{DB1} vs {DB2}\nPerformance diff")

    for ax in axes[-1, :]:
        ax.set_xlabel(f"{ch} {stat_metric} diff {DB1} and DB2")

    metric_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=metric_colors[m], markeredgecolor="black",
               markersize=8, label=metric_labels[m])
        for m in metrics
    ]
    model_handles = [
        Line2D([0], [0], marker=model_markers[m], linestyle="",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=8, label=m)
        for m in models_order
    ]

    fig.legend(handles=metric_handles, title="Metric (color)", loc="upper center",
               ncol=len(metrics), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.legend(handles=model_handles, title="Model (marker)", loc="upper center",
               ncol=min(len(models_order), 6), frameon=False, bbox_to_anchor=(0.5, 0.97))

    fig.suptitle(f"{ch}-{stat_metric}: {DB1} vs {', '.join(DB2_list)}", y=1.08, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig, plot_df

def build_plot_df(DB1, DB2_list, color_stats_df, stat_metric, *, ch, color_space):
    dfs = []
    for DB2 in DB2_list:
        color_diff_df = basedata.get_statistical_color_diff(
            color_stats_df,
            stat_metric=stat_metric,
            db_a=DB1,
            db_b=DB2
        )
        perf_diff_df = basedata.get_performance_diff(DB1, DB2)
        df = perf_diff_df.merge(color_diff_df, on="class_name", how="left")
        df["DB1"] = DB1
        df["DB2"] = DB2
        df["ch"] = ch
        df["color_space"] = color_space
        df["stat_metric"] = stat_metric
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def make_corr_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Precision_diff", "Recall_diff", "F1_diff"]

    group_cols = ["color_space", "ch", "stat_metric", "DB1", "DB2", "model"]

    rows = []
    for keys, g in plot_df.groupby(group_cols):
        *meta, model = keys

        x = g["color_diff"].to_numpy()
        for met in metrics:
            y = g[met].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            xx, yy = x[mask], y[mask]
            n = len(xx)

            if n < 3:
                pr = pp = sr = sp = np.nan
            else:
                pr, pp = pearsonr(xx, yy)
                sr, sp = spearmanr(xx, yy)

            rows.append({
                "color_space": meta[0],
                "ch": meta[1],
                "stat_metric": meta[2],
                "DB1": meta[3],
                "DB2": meta[4],
                "model": model,
                "metric": met.replace("_diff", ""),
                "n": n,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp
            })

    return pd.DataFrame(rows).sort_values(
        ["color_space", "ch", "stat_metric", "DB2", "model", "metric"]
    ).reset_index(drop=True)
