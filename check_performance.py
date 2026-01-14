import config
import pandas as pd
from sklearn.metrics import classification_report

if __name__== "__main__":

    all_rows = []

    for db, _ in config.CFG.items():
        for m in config.MODELS:
            csv_path = config.PROJECT_ROOT / f"results/{m}_{db}_REAL.csv"
            df = pd.read_csv(csv_path)

            y_true = df["label"]
            y_pred = df["preds"]

            report_dict = classification_report(
                y_true, y_pred,
                digits=3, zero_division=0,
                output_dict=True
            )

            rep_df = pd.DataFrame(report_dict).T
            rep_df.index.name = "class"
            rep_df = rep_df.reset_index()

            rep_df["model"] = m
            rep_df["db"] = db

            all_rows.append(rep_df)

    big_df = pd.concat(all_rows, ignore_index=True)

    out_path = config.PROJECT_ROOT / "results/all_performance_reports_vs_REAL.csv"
    big_df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

