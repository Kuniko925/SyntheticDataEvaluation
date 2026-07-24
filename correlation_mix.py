
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import classification_report
import config
import mappings
import data
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")

seed = 42

def perf_csv_path(model: str, db: str) -> Path:
    if db == "REAL":
        return config.PROJECT_ROOT / f"results/{model}_REAL_REAL.csv"
    ratio = db.split("_", 1)[1]  # "10"
    return config.PROJECT_ROOT / f"results/{model}_mix_{ratio}.csv"

def load_all_performance(models: list[str], dbs: list[str]) -> pd.DataFrame:
    frames = []
    for model in models:
        for db in dbs:
            df = pd.read_csv(perf_csv_path(model, db))
            frames.append(get_performance(df, DB=db, model=model))
    return pd.concat(frames, ignore_index=True)

def make_performance_diff_table(perf_df: pd.DataFrame, DB1: str) -> pd.DataFrame:
    metric_cols = ["Precision", "Recall", "F1"]

    wide = perf_df.pivot_table(
        index=["model", "label"],
        columns="DB",
        values=metric_cols
    )

    if DB1 not in wide.columns.get_level_values(1):
        raise KeyError(f"DB1 {DB1!r} not found. Available: {sorted(set(wide.columns.get_level_values(1)))}")

    base = wide.xs(DB1, level=1, axis=1)

    out = []
    for db2 in sorted(set(wide.columns.get_level_values(1))):
        if db2 == DB1:
            continue
        d = (base - wide.xs(db2, level=1, axis=1)).abs()
        d.columns = [f"{c}_diff" for c in d.columns]
        d = d.reset_index().assign(DB2=db2)
        out.append(d)

    return pd.concat(out, ignore_index=True)

def get_performance(df, DB, model):
    y_true = df["label"].to_numpy()
    y_pred = df["preds"].to_numpy()

    rep = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )

    rows = []
    for lab_str, d in rep.items():
        if not lab_str.isdigit():
            continue
        lab = int(lab_str)
        rows.append({
            "DB": DB,
            "model": model,
            "label": lab,
            "Precision": d["precision"],
            "Recall": d["recall"],
            "F1": d["f1-score"],
            "Support": d["support"],
        })
    return pd.DataFrame(rows)

def get_statistical_color_diff(df, stat_metric="mean", db_a="REAL", db_b="MIX_10"):
    g = df.groupby(["DB", "label"], as_index=False)[stat_metric].mean()
    wide = g.pivot(index="label", columns="DB", values=stat_metric)

    missing = [k for k in [db_a, db_b] if k not in wide.columns]
    if missing:
        raise KeyError(f"Missing DB in color_stats_df: {missing}. Available: {list(wide.columns)}")

    wide["color_diff"] = (wide[db_a] - wide[db_b]).abs()
    return wide.reset_index()[["label", "color_diff"]]



def channel_stats_from_pixels(pixels: np.ndarray, hist_range=(0, 256), bins=256) -> dict:
    """Compute stats from 1D pixel array."""
    # histogram/pdf for entropy
    hist, _ = np.histogram(pixels, bins=bins, range=hist_range)
    hist = hist.astype(float)
    pdf = hist / hist.sum() if hist.sum() else hist

    return {
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "skew": float(stats.skew(pixels)),
        "kurtosis": float(stats.kurtosis(pixels)),
        "entropy": float(stats.entropy(pdf)),
    }


def iter_channel_pixels_from_df(df: pd.DataFrame, color_space: str, path_col: str) :
    spec = mappings.CS[color_space]
    for p in df[path_col].astype(str):
        img = cv2.imread(p)
        if img is None:
            continue
        cimg = cv2.cvtColor(img, spec.cvt_code)
        yield cimg[:, :, spec.idx].ravel()


def compute_df_stats(df: pd.DataFrame, color_space: str, path_col: str) -> pd.DataFrame:
    pix_list = list(iter_channel_pixels_from_df(df, color_space, path_col))
    if not pix_list:
        return pd.DataFrame([])

    spec = mappings.CS[color_space]
    pixels = np.concatenate(pix_list, axis=0)
    return pd.DataFrame([channel_stats_from_pixels(pixels, hist_range=spec.hist_range)])



def create_dataframe_from_dfs(
    db_to_df: dict[str, pd.DataFrame],
    color_space: str,
    path_col: str = "filepath",
    class_col: str = "label",
) -> pd.DataFrame:
    all_rows = []
    for db, df in db_to_df.items():
        for lab, g in df.groupby(class_col):
            s = compute_df_stats(g, color_space, path_col=path_col)
            if s.empty:
                continue

            s["DB"] = db
            s["label"] = int(lab)
            s["color_space"] = color_space
            all_rows.append(s)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame([])



def build_gap_plot_df(DB1, DB2_list, perf_diff_all, color_stats_df, stat_metrics, metric_key):
    frames = []
    perf = perf_diff_all.query("DB2 in @DB2_list")

    for sm in stat_metrics:
        for db2 in DB2_list:
            cd = get_statistical_color_diff(
                color_stats_df, stat_metric=sm, db_a=DB1, db_b=db2
            )
            df = perf.query("DB2 == @db2").merge(cd, on="label", how="left")
            frames.append(df.assign(stat_metric=sm))

    plot_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return plot_df.dropna(subset=["color_diff", metric_key]) if not plot_df.empty else plot_df

def make_color_diff_table(color_stats_df: pd.DataFrame, stat_metrics: list[str]) -> pd.DataFrame:
    g = (color_stats_df
         .groupby(["DB", "label"], as_index=False)[stat_metrics]
         .mean())
    long = g.melt(id_vars=["DB","label"], value_vars=stat_metrics,
                  var_name="stat_metric", value_name="stat_value")
    return long

def color_diff_from_long(color_long: pd.DataFrame, DB1: str, DB2_list: list[str]) -> pd.DataFrame:
    base = color_long.query("DB == @DB1").rename(columns={"stat_value":"base"}).drop(columns=["DB"])
    other = color_long.query("DB in @DB2_list").rename(columns={"stat_value":"other"})
    d = other.merge(base, on=["label","stat_metric"], how="left")
    d["color_diff"] = (d["base"] - d["other"]).abs()
    return d[["DB","label","stat_metric","color_diff"]].rename(columns={"DB":"DB2"})


def make_gap_figure(
    DB1: str,
    DB2_list: list[str],
    perf_diff_all: pd.DataFrame,
    color_stats_df: pd.DataFrame,
    metric_key: str,
    stat_metrics: list[str],
    nrows: int = 2,
    ncols: int = 3,
    annot_model: str | None = "MobileNetV2",
):
    # --- build long df once ---
    plot_df = build_gap_plot_df(DB1, DB2_list, perf_diff_all, color_stats_df, stat_metrics, metric_key)

    # --- styles ---
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stat_color = {sm: cycle[i % len(cycle)] for i, sm in enumerate(stat_metrics)}

    models_order = plot_df["model"].dropna().unique().tolist()
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    model_marker = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(models_order)}

    db2_alpha = {db2: 0.60 for db2 in DB2_list}
    if DB2_list:
        db2_alpha[DB2_list[0]] = 0.90
    if len(DB2_list) > 1:
        db2_alpha[DB2_list[1]] = 0.35

    db2_edge = {db2: ("black" if i == 0 else "none") for i, db2 in enumerate(DB2_list)}
    db2_lw   = {db2: (0.5 if i == 0 else 0.0) for i, db2 in enumerate(DB2_list)}

    # --- figure ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    # global y limit
    y = plot_df[metric_key].to_numpy()
    ypad = (y.max() - y.min()) * 0.05 if y.size and y.max() > y.min() else 0.01
    ylim = (y.min() - ypad, y.max() + ypad) if y.size else (-0.01, 0.01)

    # --- draw per stat_metric ---
    for i, sm in enumerate(stat_metrics):
        ax = axes[i]
        d = plot_df[plot_df["stat_metric"] == sm]
        if d.empty:
            ax.axis("off")
            continue

        x = d["color_diff"].to_numpy()
        xpad = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 1.0
        ax.set_xlim(x.min() - xpad, x.max() + xpad)
        ax.set_ylim(*ylim)

        for (db2, model), g in d.groupby(["DB2", "model"], sort=False):
            ax.scatter(
                g["color_diff"], g[metric_key],
                color=stat_color[sm],
                marker=model_marker.get(model, "o"),
                s=60,
                alpha=db2_alpha.get(db2, 0.8),
                edgecolors=db2_edge.get(db2, "none"),
                linewidths=db2_lw.get(db2, 0.0),
            )

            if annot_model and model == annot_model:
                for _, row in g.iterrows():
                    ax.annotate(
                        str(row["label"]),
                        (row["color_diff"], row[metric_key]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=12, alpha=db2_alpha.get(db2, 0.8),
                    )

        ax.set_title(sm, fontsize=14)
        ax.grid(alpha=0.2)
        if i % ncols == 0:
            ax.set_ylabel(metric_key, fontsize=14)

    # hide unused axes
    for j in range(len(stat_metrics), nrows * ncols):
        axes[j].axis("off")

    # --- legend ---
    model_handles = [
        Line2D([0], [0], marker=model_marker[m], linestyle="", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label=m)
        for m in models_order
    ]
    db2_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="gray",
               markeredgecolor=db2_edge.get(db2, "none"),
               alpha=db2_alpha.get(db2, 0.8), markersize=8, label=db2)
        for db2 in DB2_list
    ]
    fig.legend(handles=model_handles + db2_handles,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, frameon=False)

    return fig, plot_df

def build_db_to_df(ratios: list[int]) -> dict[str, pd.DataFrame]:
    db_to_df = {}

    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset("FAKE1")
    db_to_df["REAL"] = df_train_r.copy()

    for r in ratios:
        real_ratio = r / 100.0
        db_to_df[f"MIX_{r}"] = data.make_subset(real_ratio)

    return db_to_df


def make_real_vs_mix_diff_wide_table(
    color_stats_df: pd.DataFrame,
    DB1: str,
    DB2_list: list[str],
    stat_metrics: list[str],
) -> pd.DataFrame:

    color_long = make_color_diff_table(color_stats_df, stat_metrics=stat_metrics)
    diff_long = color_diff_from_long(color_long, DB1=DB1, DB2_list=DB2_list)
    wide = (diff_long
            .pivot_table(index=["label", "stat_metric"],
                         columns="DB2",
                         values="color_diff",
                         aggfunc="mean")
            .reset_index())

    cols = ["label", "stat_metric"] + [c for c in DB2_list if c in wide.columns]
    wide = wide[cols]
    wide = wide.rename(columns={"label": "LABEL", "stat_metric": "Stats"})

    return wide

def make_f1_gap_wide_table_by_model(perf_diff_all: pd.DataFrame) -> pd.DataFrame:
    d = perf_diff_all.copy()
    d["ratio"] = d["DB2"].str.extract(r"MIX_(\d+)", expand=False).astype(int)

    wide = (d.pivot_table(
                index=["model", "label"],
                columns="ratio",
                values="F1_diff",
                aggfunc="mean"
            )
            .reset_index())

    ratio_cols = sorted([c for c in wide.columns if isinstance(c, (int, np.integer))])
    wide = wide.rename(columns={r: f"Diff_{int(r)}" for r in ratio_cols})
    wide = wide.rename(columns={"label": "LABEL"})

    wide = wide[["model", "LABEL"] + [f"Diff_{r}" for r in ratio_cols]]
    return wide



def compute_color_perf_correlation_by_model_pearson(
    color_wide: pd.DataFrame,
    f1_gap_wide: pd.DataFrame,
) -> pd.DataFrame:

    # --- color: wide -> long ---
    color_long = color_wide.melt(
        id_vars=["LABEL", "Stats"],
        var_name="DB2",
        value_name="color_diff"
    )
    color_long["ratio"] = color_long["DB2"].str.extract(r"MIX_(\d+)", expand=False).astype(int)

    # --- F1: wide -> long ---
    f1_long = f1_gap_wide.melt(
        id_vars=["model", "LABEL"],
        var_name="ratio",
        value_name="F1_diff"
    )
    f1_long["ratio"] = f1_long["ratio"].str.extract(r"Diff_(\d+)", expand=False).astype(int)

    # --- merge on LABEL + ratio ---
    merged = (f1_long.merge(color_long, on=["LABEL", "ratio"], how="inner")
                    .dropna(subset=["color_diff", "F1_diff"]))

    rows = []
    for (model, stat), sub in merged.groupby(["model", "Stats"], sort=False):
        if len(sub) < 3:
            continue
        r, p = pearsonr(sub["color_diff"], sub["F1_diff"])
        rows.append({
            "model": model,
            "Stat": stat,
            "N": len(sub),
            "Pearson_r": r,
            "Pearson_p": p,
        })

    return pd.DataFrame(rows)

# ---- Main ----
def main():
    out_dir = config.PROJECT_ROOT / "results2"
    out_dir.mkdir(parents=True, exist_ok=True)

    DB1 = "REAL"
    ratios = list(range(10, 100, 10))

    db2_list = [f"MIX_{r}" for r in ratios]
    db_list = [DB1] + db2_list

    perf_df = load_all_performance(config.MODELS, db_list)
    perf_diff_all = make_performance_diff_table(perf_df, DB1=DB1)
    db_to_df = build_db_to_df(ratios)

    f1_gap_by_model = make_f1_gap_wide_table_by_model(perf_diff_all)
    out_csv = out_dir / "F1_gap_by_model_REAL_vs_MIX.csv"
    f1_gap_by_model.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    for color_space in mappings.CS.keys():
        color_stats_df = create_dataframe_from_dfs(
            db_to_df=db_to_df,
            color_space=color_space,
            path_col="filepath",
            class_col="label",
        )

        fig, _ = make_gap_figure(
            DB1=DB1,
            DB2_list=db2_list,
            perf_diff_all=perf_diff_all,
            color_stats_df=color_stats_df,
            metric_key="F1_diff",
            stat_metrics=mappings.stat_metrics,
        )

        out_png = out_dir / f"gap_{color_space}_F1_diff_MIX.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("saved:", out_png)

        diff_wide = make_real_vs_mix_diff_wide_table(
            color_stats_df=color_stats_df,
            DB1=DB1,
            DB2_list=db2_list,
            stat_metrics=mappings.stat_metrics,
        )

        out_csv = out_dir / f"diff_table_{color_space}_REAL_vs_MIX.csv"
        diff_wide.to_csv(out_csv, index=False)
        print("saved:", out_csv)

        corr_df = compute_color_perf_correlation_by_model_pearson(
            diff_wide,
            f1_gap_by_model
        )

        out_corr = out_dir / f"correlation_{color_space}.csv"
        corr_df.to_csv(out_corr, index=False)
        print("saved:", out_corr)


if __name__ == "__main__":

    main()
