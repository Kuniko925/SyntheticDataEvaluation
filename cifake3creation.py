import pandas as pd
import config


def replace_fake6_with_real_as_fake(df: pd.DataFrame, target_label=6) -> pd.DataFrame:
    df = df.copy()

    fake6_mask = (df["label"] == target_label) & (df["rf"] == "FAKE")
    n_fake6 = int(fake6_mask.sum())

    # Source
    real6 = df[(df["label"] == target_label) & (df["rf"] == "REAL")].copy()
    if real6.empty:
        raise ValueError(f"No REAL samples found for label={target_label}")

    # Remove
    df_wo_fake6 = df.loc[~fake6_mask].copy()

    # make fake
    real6["rf"] = "FAKE"

    # merge
    out = pd.concat([df_wo_fake6, real6], ignore_index=True)

    return out

if __name__== "__main__":

    csv_path = config.PROJECT_ROOT / 'cifake3/train.csv'
    df = pd.read_csv(csv_path)
    df_new = replace_fake6_with_real_as_fake(df, target_label=0)
    df_new = replace_fake6_with_real_as_fake(df_new, target_label=3)
    df_new = replace_fake6_with_real_as_fake(df_new, target_label=4)
    df_new = replace_fake6_with_real_as_fake(df_new, target_label=5)
    df_new = replace_fake6_with_real_as_fake(df_new, target_label=6)
    df_new = replace_fake6_with_real_as_fake(df_new, target_label=7)

    csv_path = config.PROJECT_ROOT / 'cifake3/train.csv'
    df_new.to_csv(csv_path, index=False)
