from sklearn.metrics import confusion_matrix
import basedata
from scipy.stats import pearsonr, spearmanr
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_gap_figure(
    DB1: str,
    DB2_list: list[str],
    color_stats_df,
    ch: str,
    metric_key: str,          # "Precision_diff" / "Recall_diff" / "F1_diff"
    stat_metrics: list[str],  # ["mean","std","skew","kurtosis","entropy"]
    nrows: int = 2,
    ncols: int = 3,
    annot_model: str | None = "MobileNetV2",
):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stat_metric_colors = {sm: cycle[i % len(cycle)] for i, sm in enumerate(stat_metrics)}

    perf_map = {DB2: basedata.get_performance_diff(DB1, DB2) for DB2 in DB2_list}

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    tmp_models = pd.concat([perf_map[db2][["model"]] for db2 in DB2_list], ignore_index=True)["model"].unique()
    models_order = list(tmp_models)
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    model_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(models_order)}

    db2_alpha = {DB2_list[0]: 0.90}
    if len(DB2_list) > 1:
        db2_alpha[DB2_list[1]] = 0.35
    for db2 in DB2_list[2:]:
        db2_alpha[db2] = 0.60

    db2_edge = {db2: ("black" if i == 0 else "none") for i, db2 in enumerate(DB2_list)}
    db2_lw   = {db2: (0.5 if i == 0 else 0.0) for i, db2 in enumerate(DB2_list)}

    y_vals = []
    x_vals = []
    plot_dfs = {}
    for stat_metric in stat_metrics:
        dfs = []
        for DB2 in DB2_list:
            color_diff_df = basedata.get_statistical_color_diff(
                color_stats_df,
                stat_metric=stat_metric,
                db_a=DB1,
                db_b=DB2
            )
            df = perf_map[DB2].merge(color_diff_df, on="class_name", how="left")
            df["DB2"] = DB2
            dfs.append(df)

        plot_df = pd.concat(dfs, ignore_index=True)
        plot_df = plot_df.dropna(subset=["color_diff", metric_key])
        plot_dfs[stat_metric] = plot_df

        x_vals.append(plot_df["color_diff"].to_numpy())
        y_vals.append(plot_df[metric_key].to_numpy())

    y_all = np.concatenate(y_vals) if len(y_vals) else np.array([0, 1], dtype=float)
    ypad = (y_all.max() - y_all.min()) * 0.05 if y_all.max() > y_all.min() else 0.01
    ylim = (y_all.min() - ypad, y_all.max() + ypad)

    for i, stat_metric in enumerate(stat_metrics):
        ax = axes[i]
        plot_df = plot_dfs[stat_metric]

        x = plot_df["color_diff"].to_numpy()
        xpad = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 1.0
        xlim = (x.min() - xpad, x.max() + xpad)

        for DB2 in DB2_list:
            g_db = plot_df[plot_df["DB2"] == DB2]
            for model, g_m in g_db.groupby("model"):
                ax.scatter(
                    g_m["color_diff"], g_m[metric_key],
                    color=stat_metric_colors[stat_metric],
                    marker=model_markers.get(model, "o"),
                    s=60,
                    alpha=db2_alpha.get(DB2, 0.8),
                    edgecolors=db2_edge.get(DB2, "none"),
                    linewidths=db2_lw.get(DB2, 0.0),
                )

                if annot_model is not None and model == annot_model:
                    for _, row in g_m.iterrows():
                        ax.annotate(
                            str(row["class_name"]),
                            (row["color_diff"], row[metric_key]),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=12,
                            alpha=db2_alpha.get(DB2, 0.8),
                        )

        ax.set_title(stat_metric, fontsize=14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
        if i % ncols == 0:
            ax.set_ylabel(f'{metric_key}', fontsize=14)

    for j in range(len(stat_metrics), nrows * ncols):
        axes[j].axis("off")

    model_handles = [
        Line2D([0], [0], marker=model_markers[m], linestyle="",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label=m)
        for m in models_order
    ]
    db2_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="gray",
               markeredgecolor=db2_edge.get(db2, "none"),
               alpha=db2_alpha.get(db2, 0.8),
               markersize=8, label=db2)
        for db2 in DB2_list
    ]

    all_handles = model_handles + db2_handles
    fig.legend(
        handles=all_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
    )
    return fig, plot_dfs

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
