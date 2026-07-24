import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":


    dir = Path(config.PROJECT_ROOT) / "results"

    models = ["ResNet50", "MobileNetV2", "ViT16"]
    epochs = [0] + list(range(10, 150, 10))+ [150]



    rows = []

    for model in models:


        for r in epochs:

            if r == 0:
                csv_path = dir / f"{model}_FAKE1_REAL.csv"
            else:
                csv_path = dir / f"{model}_Unfreeze_{r}.csv"

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
                        "epochs": r,
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
                        "epochs": r,
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

    cls_df = cls_df.sort_values(["model", "class_id", "epochs"]).copy()
    cls_df["delta_f1"] = cls_df.groupby(["model", "class_id"])["f1"].diff()

    models = ["ResNet50", "MobileNetV2", "ViT16"]
    classes = list(range(10))
    ratios = sorted(cls_df["epochs"].unique())

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)

    for idx, c in enumerate(classes):
        ax = axes[idx // 5, idx % 5]
        ax.set_title(config.label_to_class[c], fontsize=14)
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        if idx % 5 == 4:
            ax2.set_ylabel("Improvements (%)", fontsize=14)
        else:
            ax2.set_yticklabels([])

        is_last_panel = (idx == len(classes) - 1)

        for m in models:
            dm = cls_df[(cls_df["model"] == m) & (cls_df["class_id"] == c)].copy()
            f1_series = dm.set_index("epochs")["f1"].reindex(ratios)
            d_series = dm.set_index("epochs")["delta_f1"].reindex(ratios)

            ax.plot(
                ratios,
                f1_series.values,
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=m if is_last_panel else None,
            )

            ax2.plot(
                ratios,
                d_series.values,
                marker="x",
                linewidth=1.4,
                markersize=6,
                alpha=0.6,
            )

        ax.set_xticks(ratios)
        if idx // 5 == 1:
            ax.set_xlabel("Epochs", fontsize=14)
        if idx % 5 == 0:
            ax.set_ylabel("F1", fontsize=14)

        ax.set_xticks(ratios)
        ax.set_xticklabels(ratios)

    last_ax = axes[-1, -1]
    last_ax.legend(loc="center right", frameon=False)

    plt.tight_layout()
    save_path = dir / "all_models_unfreeze_f1_lines.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", save_path)

    f1_table = (cls_df
                .pivot_table(index=["model", "class_id"], columns="epochs", values="f1", aggfunc="mean")
                .reset_index()
                .rename(columns={"class_id": "label"})
                )

    epoch_cols = [c for c in f1_table.columns if isinstance(c, (int, float))]
    f1_table = f1_table.rename(columns={e: f"Epoch_{int(e)}" for e in epoch_cols})

    ordered_epochs = sorted(epoch_cols)
    f1_table = f1_table[["model", "label"] + [f"Epoch_{int(e)}" for e in ordered_epochs]]

    out_csv = dir / "table_f1_by_model_label_epochs.csv"
    f1_table.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    f1_table_mobilenet = f1_table[f1_table["model"] == "MobileNetV2"]


    epoch_cols = [c for c in f1_table_mobilenet.columns if c.startswith("Epoch_")]
    mob_long = f1_table_mobilenet.melt(
        id_vars=["model", "label"],
        value_vars=epoch_cols,
        var_name="Epoch",
        value_name="F1"
    )
    mob_long["Epoch"] = mob_long["Epoch"].str.extract(r"Epoch_(\d+)", expand=False).astype(int)
    mob_long = mob_long.sort_values(["label", "Epoch"])

    labels = sorted(mob_long["label"].unique())
    nrows, ncols = 2, 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 7), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, lab in enumerate(labels):
        ax = axes[i]
        d = mob_long[mob_long["label"] == lab]

        x = d["Epoch"].to_numpy()
        y = d["F1"].to_numpy()

        ax.vlines(x, 0, y, linewidth=2)  # 棒
        ax.scatter(x, y, s=60)  # 点

        ax.set_title(config.label_to_class[lab], fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)


        if i // ncols == nrows - 1:
            ax.set_xlabel("Epochs", fontsize=14)

        if i % ncols == 0:
            ax.set_ylabel("F1", fontsize=14)

    for j in range(len(labels), nrows * ncols):
        axes[j].axis("off")

    fig.suptitle("MobileNetV2 Class-wise F1 over Epochs (Lollipop)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = dir / "lollipop_mobilenet_2x5.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", save_path)

    all_rows = []

    for model in models:
        csv_path = dir / f"{model}_REAL_REAL.csv"
        df = pd.read_csv(csv_path)

        rep = classification_report(
            df["label"], df["preds"],
            output_dict=True, zero_division=0
        )

        rep_df = pd.DataFrame(rep).T
        rep_df.insert(0, "model", model)
        rep_df.insert(1, "row", rep_df.index)
        rep_df = rep_df.reset_index(drop=True)

        all_rows.append(rep_df)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_csv(dir / "all_models_classification_report.csv", index=False)



