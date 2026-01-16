import pandas as pd
import colorspace
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import config
import plotting
import basedata

def create_dataframe(image_dir, class_names, color_space, ch, unit='Image'):
    all_rows = []
    for k, v in image_dir.items():
        for cls in class_names:
            datapath = config.PROJECT_ROOT / v['ImageDir'] / cls
            s_values = colorspace.get_channel_stats(datapath) if unit == "Image" else colorspace.get_channel_class_stats(datapath, color_space, ch)

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

        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), constrained_layout=True)

        cms = {}
        ims = []

        for ax, m in zip(axes, models):
            csv_path = config.build_res_filepath(data_type, m)

            cm_raw, class_names, im = plotting.plot_confusion_matrix_from_performance(
                csv_path,
                config.label_to_class,
                title=f"{m} Confusion Matrix",
                normalize=None,
                ax=ax,
                add_colorbar=False,
            )
            cms[m] = cm_raw
            ims.append(im)

        fig.colorbar(ims[0], ax=axes, shrink=0.85)
        out_path = config.PROJECT_ROOT / f"results/class_heatmap_{data_type}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    # Gpa scatterplot between Color statis and performance
    DB1 = "REAL"
    db2_list = ["FAKE1", "FAKE2"]
    stat_metrics = ["mean", "std", "skew", "kurtosis", "entropy"]

    targets = [
        ("LAB", "L"),
        ("HSV", "S"),
        ("HSV", "V"),
        ("YCRCB", "Y"),
    ]

    for color_space, ch in targets:
        color_stats_df = create_dataframe(
            config.CFG,
            list(config.label_to_class.values()),
            color_space=color_space,
            ch=ch,
            unit="Data"
        )

        out_pdf = config.PROJECT_ROOT / "results" / f"gap_{color_space}_{ch}_{DB1}_FAKE1_FAKE2_by_metric_pages.pdf"

        with PdfPages(out_pdf) as pdf:
            for stat_metric in stat_metrics:
                fig, _ = plotting.make_gap_figure(
                    DB1=DB1,
                    DB2_list=db2_list,
                    color_stats_df=color_stats_df,
                    ch=ch,
                    stat_metric=stat_metric,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print("saved:", out_pdf)

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
        out_path = config.PROJECT_ROOT / "results" / f"corr_ALL_REAL_{db2}_heatmap.png"
        d = corr_all_df.query(f"DB2 == '{db2}'")

        mat = d.pivot_table(
            index=["model", "metric"],
            columns=["color_space", "ch", "stat_metric"],
            values="pearson_r"
        ).sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation (Pearson): REAL vs {db2}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
