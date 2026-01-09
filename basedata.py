from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import config

def get_performance_data(DB, model):
    csv_path = config.PROJECT_ROOT / config.build_res_filepath(DB, model)
    df = pd.read_csv(csv_path)
    return df.assign(DB=DB, model=model)

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
            "class_name": config.label_to_class[lab],
            "Precision": d["precision"],
            "Recall": d["recall"],
            "F1": d["f1-score"],
            "Support": d["support"],
        })
    return pd.DataFrame(rows)

def get_statistical_color_diff(df, stat_metric="mean", db_a="REAL", db_b="FAKE1"):
    g = df.groupby(["DB", "class_name"], as_index=False)[stat_metric].mean()
    wide = g.pivot(index="class_name", columns="DB", values=stat_metric)
    wide["color_diff"] = (wide[db_a] - wide[db_b]).abs()
    return wide.reset_index()[["class_name", "color_diff"]]

def get_performance_diff(DB1, DB2):
    dfs = []
    for data_type in list(config.CFG.keys()):
        for m in config.MODELS:
            data_df = get_performance_data(data_type, m)
            dfs.append(get_performance(data_df, data_type, m))

    perform_df = pd.concat(dfs, ignore_index=True)
    metric_cols = ["Precision", "Recall", "F1"]
    diff = perform_df.pivot_table(
        index=["model", "label", "class_name"],
        columns="DB",
        values=metric_cols
    )

    diff = np.abs(diff.xs(DB1, level=1, axis=1) - diff.xs(DB2, level=1, axis=1))
    diff.columns = [f"{c}_diff" for c in diff.columns]
    return diff.reset_index()






