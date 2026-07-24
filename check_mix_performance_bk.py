import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_diff_series(diff_long: pd.DataFrame, class_id: int, ratios: list[int]) -> pd.Series:
    # diff_long: columns = ["LABEL","MIX","DIFF"]
    s = (diff_long[diff_long["LABEL"] == class_id]
         .set_index("MIX")["DIFF"]
         .reindex(ratios))
    return s

if __name__== "__main__":

    dir = Path(config.PROJECT_ROOT) / "results2"
    cs_csv_path = dir / "diff_table_LAB_REAL_vs_MIX.csv"
    df_lab = pd.read_csv(cs_csv_path)
    cs_csv_path = dir / "diff_table_HSV_REAL_vs_MIX.csv"
    df_hsv = pd.read_csv(cs_csv_path)
    cs_csv_path = dir / "diff_table_YCRCB_REAL_vs_MIX.csv"
    df_ycrcb = pd.read_csv(cs_csv_path)

    df_lab = df_lab[df_lab["Stats"] == 'entropy']
    df_hsv = df_hsv[df_hsv["Stats"] == 'entropy']
    df_ycrcb = df_ycrcb[df_ycrcb["Stats"] == 'entropy']

    def melt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        mix_cols = [c for c in df.columns if c.startswith("MIX_")]
        long_df = (df.melt(id_vars=["LABEL"], value_vars=mix_cols,
                     var_name="MIX", value_name="DIFF"))
        long_df["MIX"] = long_df["MIX"].str.replace("MIX_", "", regex=False).astype(int)
        return long_df

    df_lab = melt_dataframe(df_lab)
    df_hsv = melt_dataframe(df_hsv)
    df_ycrcb = melt_dataframe(df_ycrcb)

    all_diffs = pd.concat([
        df_lab[["DIFF"]],
        df_hsv[["DIFF"]],
        df_ycrcb[["DIFF"]],
    ], ignore_index=True)

    max_diff = all_diffs["DIFF"].max()
    upper = max_diff * 1.05

    cs_color = {
        "LAB": "tab:purple",
        "HSV": "tab:gray",
        "YCrCb": "tab:olive",
    }

    dir = Path(config.PROJECT_ROOT) / "results"

    models = ["ResNet50", "MobileNetV2", "ViT16"]
    ratios = [0] + list(range(10, 100, 10)) + [100]

    rows = []

    for model in models:
        for r in ratios:

            if r == 0:
                csv_path = dir / f"{model}_FAKE1_REAL.csv"
            elif r == 100:
                csv_path = dir / f"{model}_REAL_REAL.csv"
            else:
                csv_path = dir / f"{model}_mix_{r}.csv"

            df = pd.read_csv(csv_path)

            rep = classification_report(
                df["label"], df["preds"],
                output_dict=True,
                zero_division=0
            )

            for k, v in rep.items():
                if k == "accuracy":
                    rows.append({
                        "model": model,
                        "mix_ratio": r,
                        "section": "accuracy",
                        "target": "accuracy",
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "support": None,
                        "value": v
                    })
                else:
                    rows.append({
                        "model": model,
                        "mix_ratio": r,
                        "section": "class" if str(k).isdigit() else "avg",
                        "target": k,
                        "precision": v.get("precision"),
                        "recall": v.get("recall"),
                        "f1": v.get("f1-score"),
                        "support": v.get("support"),
                        "value": None
                    })

    report_df = pd.DataFrame(rows)

    cls_df = report_df[report_df["section"] == "class"].copy()
    cls_df = cls_df[cls_df["target"].astype(str).str.match(r"^\d+$")]
    cls_df["class_id"] = cls_df["target"].astype(int)


    models = ["ResNet50", "MobileNetV2", "ViT16"]
    classes = list(range(10))

    last_ax_ref = None
    last_ax2_ref = None

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)

    for idx, c in enumerate(classes):
        ax = axes[idx // 5, idx % 5]
        ax.set_title(config.label_to_class[c], fontsize=14)
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        is_right_col = (idx % 5 == 4)
        if not is_right_col:
            ax2.tick_params(right=False, labelright=False)
            ax2.spines["right"].set_visible(False)
        else:
            ax2.tick_params(right=True, labelright=True)
            ax2.set_ylabel("Entropy diff vs REAL", fontsize=14)

        is_last_panel = (idx == len(classes) - 1)
        if is_last_panel:
            last_ax_ref = ax
            last_ax2_ref = ax2

        for m in models:
            dm = cls_df[(cls_df["model"] == m) & (cls_df["class_id"] == c)].copy()
            f1_series = dm.set_index("mix_ratio")["f1"].reindex(ratios)


            ax.plot(
                ratios,
                f1_series.values,
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=m if is_last_panel else None,
            )

        lab_s = get_diff_series(df_lab, c, ratios)
        hsv_s = get_diff_series(df_hsv, c, ratios)
        ycrcb_s = get_diff_series(df_ycrcb, c, ratios)

        ax2.plot(ratios, lab_s.values, linestyle="--", marker="x", linewidth=1.2,  color=cs_color["LAB"], label=None)
        ax2.plot(ratios, hsv_s.values, linestyle="--", marker="x", linewidth=1.2,  color=cs_color["HSV"], label=None)
        ax2.plot(ratios, ycrcb_s.values, linestyle="--", marker="x", linewidth=1.2,  color=cs_color["YCrCb"], label=None)

        ax2.set_ylim(0, upper)

        ax.set_xticks(ratios)
        if idx // 5 == 1:
            ax.set_xlabel("REAL Ratio (%)", fontsize=14)
        if idx % 5 == 0:
            ax.set_ylabel("F1", fontsize=14)
        if idx % 5 == 4:
            ax2.set_ylabel("Entropy diff vs REAL", fontsize=14)

    cs_handles = [
        Line2D([0], [0], linestyle="--", marker="x", color=cs_color["LAB"], label="LAB"),
        Line2D([0], [0], linestyle="--", marker="x", color=cs_color["HSV"], label="HSV"),
        Line2D([0], [0], linestyle="--", marker="x", color=cs_color["YCrCb"], label="YCrCb"),
    ]
    cs_labels = ["LAB", "HSV", "YCrCb"]
    model_handles, model_labels = last_ax_ref.get_legend_handles_labels()

    fig.subplots_adjust(right=0.80)

    leg1 = fig.legend(
        model_handles, model_labels,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.95),
        frameon=False
    )

    leg2 = fig.legend(
        cs_handles, cs_labels,
        title="Color space (diff)",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.60),
        frameon=False
    )

    fig.add_artist(leg1)
    fig.add_artist(leg2)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_path = dir / "all_models_classwise_f1_lines.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", save_path)


