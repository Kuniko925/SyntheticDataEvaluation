from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import colorspace
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent # Path to folder opened files


def plot_dumbbell(df, metrics, savepath):

    #=========================================================
    # ダンベルプロット描画関数（1 axes に FAKE1 & FAKE2 を重ねる）
    #=========================================================
    def dumbbell(ax, subdf, metric):

        classes = sorted(subdf["class_name"].unique())

        # ========================================
        # REAL の値の最小値・最大値を取得して帯を描く
        # ========================================
        real_vals = subdf[subdf["image_type"] == "REAL"][metric].values
        if len(real_vals) > 0:
            real_min = real_vals.min()
            real_max = real_vals.max()
            ax.axvspan(real_min, real_max, color="red", alpha=0.1, label="_nolegend_")

        for cls in classes:

            row_real  = subdf[(subdf["class_name"] == cls) & (subdf["image_type"] == "REAL")]
            row_fake1 = subdf[(subdf["class_name"] == cls) & (subdf["image_type"] == "FAKE1")]
            row_fake2 = subdf[(subdf["class_name"] == cls) & (subdf["image_type"] == "FAKE2")]

            if row_real.empty:
                continue

            y = classes.index(cls)
            real = row_real[metric].values[0]

            # --- FAKE1 vs REAL ---
            if not row_fake1.empty:
                fake1 = row_fake1[metric].values[0]
                ax.plot([real, fake1], [y, y], color="gray", alpha=0.6)
                ax.scatter(real,  y, color="red", s=100, label="REAL" if y == 0 else "")
                ax.scatter(fake1, y, color="blue", s=100, label="FAKE1" if y == 0 else "")

            # --- FAKE2 vs REAL ---
            if not row_fake2.empty:
                fake2 = row_fake2[metric].values[0]
                ax.plot([real, fake2], [y, y], color="gray", alpha=0.6)
                ax.scatter(fake2, y, color="green", s=100, label="FAKE2" if y == 0 else "")


        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title(f"{metric} comparison")
        ax.grid(alpha=0.3)


    #=========================================================
    # 図のレイアウト：3列 × 6行（18 metrics）
    #=========================================================
    rows = 6
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(22, 30))
    axes = axes.flatten()

    #=========================================================
    # 各 metric を順番に描画
    #=========================================================
    for i, metric in enumerate(metrics):
        ax = axes[i]
        dumbbell(ax, df, metric)

    # 空の軸があれば非表示（metrics が 18 のときは不要）
    for j in range(len(metrics), rows * cols):
        axes[j].axis("off")

    #=========================================================
    # Legend を1箇所にまとめて表示
    #=========================================================
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # legend 位置確保
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def colorspace_basic_stats(folders, class_names, savepath=None):
    rows = []
    for folder in folders:
        for class_name in class_names:
            data = colorspace.load_color_images(PROJECT_ROOT / folder / class_name, "LAB")
            l_stats = colorspace.l_stats(data, ch="L")
            ab_stats = colorspace.ab_stats(data, ch1="a", ch2="b")

            data = colorspace.load_color_images(PROJECT_ROOT / folder / class_name, "HSV")
            s_stats = colorspace.l_stats(data, ch="S")
            v_stats = colorspace.l_stats(data, ch="V")

            data = colorspace.load_color_images(PROJECT_ROOT / folder / class_name, "YCrCb")
            y_stats = colorspace.l_stats(data, ch="Y")
            cbcr_stats = colorspace.ab_stats(data, ch1="Cr", ch2="Cb")

            if folder == 'cifake1/train/FAKE':
                image_type = 'FAKE1'
            elif folder == 'cifake1/train/REAL':
                image_type = 'REAL'
            elif folder == 'cifake2/train/FAKE':
                image_type = 'FAKE2'
            else:
                raise ValueError("Wrong image type")

            rows.append({
                "image_type": image_type,
                "class_name": class_name,
                "L_mean": l_stats["mean"],
                "L_std": l_stats["std"],
                "L_skew": l_stats["skew"],
                "var_a": ab_stats["var_a"],
                "var_b": ab_stats["var_b"],
                "cov_ab": ab_stats["cov_ab"],
                "S_mean": s_stats["mean"],
                "S_std": s_stats["std"],
                "S_skew": s_stats["skew"],
                "V_mean": v_stats["mean"],
                "V_std": v_stats["std"],
                "V_skew": v_stats["skew"],
                "Y_mean": y_stats["mean"],
                "Y_std": y_stats["std"],
                "Y_skew": y_stats["skew"],
                "var_Cr": cbcr_stats["var_Cr"],
                "var_Cb": cbcr_stats["var_Cb"],
                "cov_CrCb": cbcr_stats["cov_CrCb"],
            })



    df = pd.DataFrame(rows)

    metrics = ["L_mean", "L_std", "L_skew", "var_a", "var_b", "cov_ab", "S_mean", "S_std", "S_skew", "V_mean", "V_std", "V_skew", "Y_mean", "Y_std", "Y_skew", "var_Cr", "var_Cb", "cov_CrCb"]
    plot_dumbbell(df, metrics, savepath)
    return df

label_to_class = {0: 'airplane',
                      1: 'automobile',
                      2: 'bird',
                      3: 'cat',
                      4: 'deer',
                      5: 'dog',
                      6: 'frog',
                      7: 'horse',
                      8: 'ship',
                      9: 'truck'}

def get_performance_data(res_paths):

    dfs = []
    for res_path in res_paths:
        p = PROJECT_ROOT / res_path
        tmp = pd.read_csv(p)

        tmp['dataset'] = res_path.split('/')[0]
        tmp["model"] = res_path.split('/')[2].split('_')[0]
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    df["class_name"] = df["label"].map(label_to_class)

    rows = []
    for (model, class_name), g in df.groupby(["model", "class_name"]):
        y_true = g["label"]
        y_pred = g["preds"]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        rows.append({
            "model": model,
            "label": g["label"].iloc[0],
            "class_name": class_name,
            "Accuracy": acc,
            "F1": f1,
            "support": len(g)
        })

    return pd.DataFrame(rows)

def save_correlation_heatmap(stat_df, perform_df, savepath):

    models = perform_df["model"].unique()
    class_names = list(label_to_class.values())

    disp_cols = [
        "L_mean", "L_std", "L_skew",
        "var_a", "var_b", "cov_ab",
        "S_mean", "S_std", "S_skew",
        "V_mean", "V_std", "V_skew",
        "Y_mean", "Y_std", "Y_skew",
        "var_Cr", "var_Cb", "cov_CrCb", "Accuracy", "F1"
    ]

    perf_cols = ["Accuracy", "F1"]

    fig, axes = plt.subplots(5, 6, figsize=(30, 24))
    idx = 0

    for model in models:
        for class_name in class_names:
            ax = axes[idx // 6, idx % 6]

            stat_subset = stat_df[stat_df["class_name"] == class_name].copy()
            perform_subset = perform_df[(perform_df["model"] == model) & (perform_df['class_name'] == class_name)].copy()
            subset = pd.merge(stat_subset, perform_subset, on="class_name")
            corr_all = subset[disp_cols].corr()
            print(corr_all)
            corr_df = corr_all.loc[perf_cols]

            sns.heatmap(
                corr_df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar=False,
                ax=ax
            )
            ax.set_title(f"{model} | {class_names}", fontsize=10)
            idx += 1

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {savepath}")


if __name__== "__main__":

    folders = ["cifake1/train/REAL", "cifake1/train/FAKE", "cifake2/train/FAKE"]
    class_names = list(label_to_class.values())

    savepath = PROJECT_ROOT / "results/cielab.pdf"
    df = colorspace_basic_stats(folders, class_names, savepath)

    res_paths = ['cifake1/results/ResNet50Model_fake_real.csv',
                 'cifake1/results/MobileNetV2_fake_real.csv',
                 'cifake1/results/ViT16_fake_real.csv']

    fake1_df = get_performance_data(res_paths)

    res_paths = ['cifake1/results/ResNet50Model_real_real.csv',
                 'cifake1/results/MobileNetV2_real_real.csv',
                 'cifake1/results/ViT16_real_real.csv']

    real_df = get_performance_data(res_paths)

    res_paths = ['cifake2/results/ResNet50Model_fake_real.csv',
                 'cifake2/results/MobileNetV2_fake_real.csv',
                 'cifake2/results/ViT16_fake_real.csv']

    fake2_df = get_performance_data(res_paths)

    savepath = PROJECT_ROOT / "results/fake1_correlation_heatmap.pdf"
    save_correlation_heatmap(df, fake1_df, savepath)

    #savepath = PROJECT_ROOT / "results/real_correlation_heatmap.pdf"
    #save_correlation_heatmap(df, real_df, savepath)

    #savepath = PROJECT_ROOT / "results/fake2_correlation_heatmap.pdf"
    #save_correlation_heatmap(df, fake2_df, savepath)




