import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__== "__main__":



    dir = Path(config.PROJECT_ROOT) / "results"

    models = ["ResNet50", "MobileNetV2", "ViT16"]
    ratios = list(range(10, 100, 10))

    rows = []

    for model in models:
        for r in ratios:
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

    out_path = dir / "mix_classification_reports.csv"
    report_df.to_csv(out_path, index=False)

    print(report_df.head())
    print("saved:", out_path)

    cls_df = report_df[report_df["section"] == "class"].copy()
    cls_df = cls_df[cls_df["target"].astype(str).str.match(r"^\d+$")]  # "0".."9"
    cls_df["class_id"] = cls_df["target"].astype(int)

    models = cls_df["model"].unique()
    classes = list(range(10))  # 0..9

    for model in models:
        dm = cls_df[cls_df["model"] == model]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
        fig.suptitle(f"Class-wise F1 vs MIX RATIO - {model}", fontsize=14)

        for idx, c in enumerate(classes):
            ax = axes[idx // 5, idx % 5]
            dc = dm[dm["class_id"] == c].sort_values("mix_ratio")

            ax.scatter(dc["mix_ratio"], dc["f1"])
            ax.set_title(f"Class {c}")
            ax.set_ylim(0, 1)

            if idx // 5 == 1:
                ax.set_xlabel("MIX RATIO (%)")
            if idx % 5 == 0:
                ax.set_ylabel("F1")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = dir / f"{model}_classwise_f1_scatter.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
