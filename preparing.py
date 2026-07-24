import os
import pandas as pd
import config


class CIFAKEDataSet:
    def __init__(self, real_root, fake1_root, fake2_root):
        self.real_root = real_root
        self.fake1_root = fake1_root
        self.fake2_root = fake2_root
        self.class_to_label = config.make_class_to_label()

    def create_dataset(self, d_type, save_filepath=None):
        allfiles = []

        dataset_sources = [
            {
                "root": self.real_root,
                "folder": "REAL",
                "rf": "REAL",
                "real": 1,
            },
            {
                "root": self.fake1_root,
                "folder": "FAKE",
                "rf": "FAKE1",
                "real": 0,
            },
            {
                "root": self.fake2_root,
                "folder": "FAKE",
                "rf": "FAKE2",
                "real": 0,
            },
        ]

        for source in dataset_sources:
            data_dir = os.path.join(
                source["root"],
                d_type,
                source["folder"],
            )

            class_dirs = os.listdir(data_dir)

            for class_name in class_dirs:
                class_path = os.path.join(
                    data_dir,
                    class_name,
                )

                if not os.path.isdir(class_path):
                    continue

                files = os.listdir(class_path)

                for filename in files:
                    filepath = os.path.join(
                        class_path,
                        filename,
                    )

                    if not os.path.isfile(filepath):
                        continue

                    allfiles.append({
                        "filepath": filepath,
                        "class name": class_name,
                        "rf": source["rf"],
                        "real": source["real"],
                    })

        df = pd.DataFrame(allfiles)

        df["label"] = df["class name"].map(
            self.class_to_label
        )

        if df["label"].isna().any():
            unknown_classes = (
                df.loc[df["label"].isna(), "class name"]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Unknown class names: {unknown_classes}"
            )

        df["real"] = df["real"].astype(int)
        df["label"] = df["label"].astype(int)

        if save_filepath is not None:
            df.to_csv(save_filepath, index=False)

        return df


def main():
    dataset = CIFAKEDataSet(
        real_root=config.PROJECT_ROOT / "cifake1",
        fake1_root=config.PROJECT_ROOT / "cifake1",
        fake2_root=config.PROJECT_ROOT / "cifake2",
    )
    dataset.create_dataset(
        d_type="train",
        save_filepath=config.PROJECT_ROOT / "train.csv",
    )

    dataset.create_dataset(
        d_type="test",
        save_filepath=config.PROJECT_ROOT / "test.csv",
    )

if __name__== "__main__":
    main()