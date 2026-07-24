import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__== "__main__":

    dir = Path(config.PROJECT_ROOT) / "results"
    models = ["ResNet50", "MobileNetV2", "ViT16"]
    rows = []


    def add_report_rows(model: str, dataset: str, rep: dict, rows: list):
        for k, v in rep.items():
            if k == "accuracy":
                rows.append({
                    "model": model,
                    "dataset": dataset,
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
                    "dataset": dataset,
                    "section": "class" if str(k).isdigit() else "avg",
                    "target": k,
                    "precision": v.get("precision"),
                    "recall": v.get("recall"),
                    "f1": v.get("f1-score"),
                    "support": v.get("support"),
                    "value": None
                })


    for model in models:
        # MIX_REAL
        mix_path = dir / f"{model}_MIX_REAL.csv"
        mix_df = pd.read_csv(mix_path)
        mix_rep = classification_report(
            mix_df["label"], mix_df["preds"],
            output_dict=True, zero_division=0
        )
        add_report_rows(model, "MIX_REAL", mix_rep, rows)

        # FAKE1_REAL
        fake1_path = dir / f"{model}_FAKE1_REAL.csv"
        fake1_df = pd.read_csv(fake1_path)
        fake1_rep = classification_report(
            fake1_df["label"], fake1_df["preds"],
            output_dict=True, zero_division=0
        )
        add_report_rows(model, "FAKE1_REAL", fake1_rep, rows)

    report_df = pd.DataFrame(rows)

    out_path = dir / "one_class_mix_and_fake1_classification_reports.csv"
    report_df.to_csv(out_path, index=False)

    print(report_df.head())
    print("saved:", out_path)
