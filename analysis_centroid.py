from __future__ import annotations
import pandas as pd
import numpy as np
import config
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


if __name__== "__main__":

    model_names = ['CLIP', 'DINOv2', 'DINOv3']
    reducer_names = ['UMAP', 'TSNE']
    DB = ['FAKE1', 'FAKE2']

    dfs = []
    for db in DB:
        for model in model_names:
            for reducer in reducer_names:
                fname = f"dis_{db}_{model}_{reducer}.csv"
                path = config.PROJECT_ROOT / "results" / fname

                df = pd.read_csv(path)
                df["DB"] = db
                df["model"] = model
                df["reducer"] = reducer

                dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    df = all_df.copy()

    df["centroid_dist"] = np.hypot(df["r centroid x"] - df["f centroid x"],
                                   df["r centroid y"] - df["f centroid y"])

    df["overlap_ratio"] = df["centroid_dist"] / (df["r radius"] + 1e-9)
    save_path = config.PROJECT_ROOT / f"results/distance_from_centroid.csv"
    df.to_csv(save_path, index=False)

