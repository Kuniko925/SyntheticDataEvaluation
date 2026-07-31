import pandas as pd
import matplotlib.pyplot as plt
import config
import plotting
import scipy.stats as stats
from pathlib import Path
import os
import cv2
import numpy as np

def load_color_images(dir_path, colorspace="LAB", ch='L'):

    cvt, ch_to_idx = _CS_MAP[colorspace]
    idx = ch_to_idx[ch]
    data = []
    cls = Path(dir_path).name

    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        cimg = cv2.cvtColor(img, cvt)

        data.append({
            "class": cls,
            "ch": ch,
            ch: cimg[:, :, idx].astype(np.float32),
        })

    return data

_CS_MAP = {
    "LAB":   (cv2.COLOR_BGR2LAB,   {"L": 0, "A": 1, "B": 2}),
    "HSV":   (cv2.COLOR_BGR2HSV,   {"H": 0, "S": 1, "V": 2}),
    "YCRCB": (cv2.COLOR_BGR2YCrCb, {"Y": 0, "CR": 1, "CB": 2}),
}

def get_channel_class_stats(dir_path, color_space, ch):
    data = load_color_images(dir_path, colorspace=color_space, ch=ch)
    c_data = np.concatenate([d[ch].ravel() for d in data])
    bins = np.linspace(0, 1, 256)
    hist_range = (bins[0], bins[-1])
    hist, _ = np.histogram(c_data, bins=bins, range=hist_range)
    hist = hist.astype(float)
    pdf = hist / hist.sum() if hist.sum() else hist
    return {
        "mean": c_data.mean(),
        "std": c_data.std(),
        "skew": stats.skew(c_data),
        "kurtosis": stats.kurtosis(c_data),
        "entropy": stats.entropy(pdf),
    }


def get_channel_stats(dir_path, color_space='LAB', ch='L'):
    cvt, keys = _CS_MAP[color_space]
    cidx = keys.index(ch)

    data = []
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img_bgr = cv2.imread(img_path)
        img_cvt = cv2.cvtColor(img_bgr, cvt)
        C = img_cvt[:, :, cidx]
        pixels = C.ravel()

        hist = np.bincount(pixels, minlength=256).astype(float)
        pdf = hist / hist.sum()
        data.append({
                "mean": pixels.mean(),
                "std": pixels.std(),
                "skew": stats.skew(pixels),
                "entropy": stats.entropy(pdf),
            })
    return data


def create_dataframe(image_dir, class_names, color_space, ch, unit='Image'):
    all_rows = []
    for k, v in image_dir.items():
        for cls in class_names:
            datapath = config.PROJECT_ROOT / v['ImageDir'] / cls
            s_values = get_channel_stats(datapath) if unit == "Image" else get_channel_class_stats(datapath, color_space, ch)

            if isinstance(s_values, dict):
                df_cls = pd.DataFrame([s_values])
            else:
                df_cls = pd.DataFrame(s_values)

            df_cls["DB"] = k
            df_cls["class_name"] = cls
            all_rows.append(df_cls)

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


if __name__== "__main__":

    models = config.MODELS

    # Plot Confusion Matrix
    for data_type in list(config.CFG.keys()):

        fig, axes = plt.subplots(
            1, len(models),
            figsize=(5 * len(models), 6),
            constrained_layout=True,
            sharey=True,
        )

        cms, ims = {}, []

        for i, (ax, m) in enumerate(zip(axes, models)):
            csv_path = config.build_res_filepath(data_type, m)
            cm_raw, class_names, im = plotting.plot_confusion_matrix_from_performance(
                csv_path,
                config.label_to_class,
                title=f"{m}",
                normalize=None,
                ax=ax,
                add_colorbar=False,
            )
            cms[m] = cm_raw
            ims.append(im)

            if i != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

        out_path = config.PROJECT_ROOT / f"results/class_heatmap_{data_type}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    # Gpa scatterplot between Color statis and performance
    DB1 = "REAL"
    db2_list = ["FAKE1", "FAKE2"]
    stat_metrics = ["mean", "std", "skew", "kurtosis", "entropy"]
    targets = [("LAB", "L"), ("HSV", "V"), ("YCRCB", "Y"),]

    out_dir = config.PROJECT_ROOT / "results"

    for color_space, ch in targets:
        color_stats_df = create_dataframe(
            config.CFG,
            list(config.label_to_class.values()),
            color_space=color_space,
            ch=ch,
            unit="Data"
        )

        fig, _ = plotting.make_gap_figure(
            DB1=DB1,
            DB2_list=db2_list,
            color_stats_df=color_stats_df,
            ch=ch,
            metric_key = 'F1_diff',
            stat_metrics=stat_metrics,
        )

        out_png = out_dir / f"gap_{ch}_F1_diff_FAKE1_FAKE2.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved:", out_png)

    # Correlation between Color statistics and performance --> DataFrame
    all_corr = []
    for color_space, ch in targets:
        color_stats_df = create_dataframe(
            config.CFG,
            list(config.label_to_class.values()),
            color_space=color_space,
            ch=ch,
            unit="Data"
        )

        for stat_metric in stat_metrics:
            plot_df = plotting.build_plot_df(
                DB1, db2_list, color_stats_df, stat_metric,
                ch=ch, color_space=color_space
            )
            corr_df = plotting.make_corr_table(plot_df)
            all_corr.append(corr_df)

    corr_all_df = pd.concat(all_corr, ignore_index=True)
    out_csv = config.PROJECT_ROOT / "results" / f"corr_ALL_REAL_FAKE1_FAKE2.csv"
    corr_all_df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    #  Visualise correlation using Heatmap
    for db2 in db2_list:
        d = corr_all_df.query(f"DB2 == '{db2}'")

        mat = d.pivot_table(
            index=["model", "metric"],
            columns=["color_space", "ch", "stat_metric"],
            values="pearson_r"
        ).sort_index(axis=1)

        out_csv = config.PROJECT_ROOT / "results" / f"corr_ALL_REAL_{db2}_heatmap.csv"
        mat.to_csv(out_csv, index=True)
        print("saved:", out_csv)


