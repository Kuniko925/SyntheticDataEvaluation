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
        csv_path = dir / f"{model}_MIX_REAL.csv"
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
                    "section": "class" if str(k).isdigit() else "avg",
                    "target": k,
                    "precision": v.get("precision"),
                    "recall": v.get("recall"),
                    "f1": v.get("f1-score"),
                    "support": v.get("support"),
                    "value": None
                })

    report_df = pd.DataFrame(rows)

    out_path = dir / "one_class_mix_classification_reports.csv"
    report_df.to_csv(out_path, index=False)

    print(report_df.head())
    print("saved:", out_path)