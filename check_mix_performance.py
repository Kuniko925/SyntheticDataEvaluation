import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__== "__main__":


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

    # improvement ratio
    cls_df = cls_df.sort_values(["model", "class_id", "mix_ratio"]).copy()
    cls_df["delta_f1"] = cls_df.groupby(["model", "class_id"])["f1"].diff()


    models = ["ResNet50", "MobileNetV2", "ViT16"]
    classes = list(range(10))
    ratios = sorted(cls_df["mix_ratio"].unique())

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)

    for idx, c in enumerate(classes):
        ax = axes[idx // 5, idx % 5]
        ax.set_title(config.label_to_class[c], fontsize=14)
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        if idx % 5 == 4:
            ax2.set_ylabel("Improvement (%)", fontsize=14)
        else:
            ax2.set_yticklabels([])

        is_last_panel = (idx == len(classes) - 1)

        for m in models:
            dm = cls_df[(cls_df["model"] == m) & (cls_df["class_id"] == c)].copy()
            f1_series = dm.set_index("mix_ratio")["f1"].reindex(ratios)
            d_series = dm.set_index("mix_ratio")["delta_f1"].reindex(ratios)

            ax.plot(
                ratios,
                f1_series.values,
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=m if is_last_panel else None,
            )

            ax2.plot(ratios, d_series.values, marker="x", linewidth=1.4, markersize=6, alpha=0.6)

        ax.set_xticks(ratios)
        if idx // 5 == 1:
            ax.set_xlabel("REAL Ratio (%)", fontsize=14)
        if idx % 5 == 0:
            ax.set_ylabel("F1", fontsize=14)

    last_ax = axes[-1, -1]
    last_ax.legend(loc="center right", frameon=False)

    plt.tight_layout()
    save_path = dir / "all_models_classwise_f1_lines.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", save_path)


