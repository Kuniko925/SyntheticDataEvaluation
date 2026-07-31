import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap
from data import TransDataset
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPModel
import config
import time

model_specs = [
    ("DINOv2", "facebook/dinov2-small"),
    ("DINOv3", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    ("CLIP", "openai/clip-vit-base-patch32"),
]



device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def reduce_dim(df_train, image_embeddings, reducer, model_name, reducer_name, db='FAKE1', seed=42):
    embeddings_2d = reducer.fit_transform(image_embeddings)
    df_train['embeddings x'] = embeddings_2d[:,0]
    df_train['embeddings y'] = embeddings_2d[:,1]
    save_filepath = config.PROJECT_ROOT / f'results/embed_{db}_{model_name}_{reducer_name}_{seed}.csv'
    df_train.to_csv(save_filepath, index=False)
    return embeddings_2d


def plot_2d_embeddings(embeddings_2d, labels, reals, model_name, reducer_name, db='FAKE1', seed=42):

    save_filepath = config.PROJECT_ROOT / f'results/{db}_{model_name}_{reducer_name}_{seed}.png'

    color_map = {0: "lightpink", 1: "lightblue"}
    rf_label_map = {0: "GEN", 1: "REAL"}
    centroid_color_map = {0: "red", 1: "blue"}

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(len(config.label_to_class)):
        ax = axes[i]
        area_dict = {}

        for rf in [1, 0]:  # "0:FAKE", "1:REAL"
            subset = (labels == i) & (reals == rf)
            ax.scatter(
                embeddings_2d[subset, 0],
                embeddings_2d[subset, 1],
                label=rf_label_map[rf],
                color=color_map[rf],
                alpha=0.3,
                s=15
            )

            # centroid
            if subset.sum() > 0:
                centroid = embeddings_2d[subset].mean(axis=0)
                ax.scatter(
                    centroid[0], centroid[1],
                    color=centroid_color_map[rf],
                    marker='X',
                    s=100,
                    linewidth=0.8,
                    label=f"{rf_label_map[rf]} Centroid"
                )

                # radius: mean point of the mean point clouds
                dists = ((embeddings_2d[subset] - centroid) ** 2).sum(axis=1) ** 0.5
                radius = dists.mean()
                area_dict[rf] = np.pi * (radius ** 2)  # area

                # circle
                circle = patches.Circle(
                    (centroid[0], centroid[1]),
                    radius,
                    color=centroid_color_map[rf],
                    alpha=0.2,
                    linestyle='--',
                    linewidth=1,
                    fill=True
                )
                ax.add_patch(circle)

        ax.set_title(f"{config.label_to_class[i]}", fontsize=16)
        ax.grid(True)

    last_ax = axes[len(config.label_to_class) - 1]
    handles, labels_ = last_ax.get_legend_handles_labels()
    last_ax.legend(handles, labels_, fontsize=14, loc="best")

    plt.tight_layout()
    plt.savefig(save_filepath)
    plt.close(fig)


def save_distance(embeddings_2d, labels, reals, model_name, reducer_name, db='FAKE1', seed=42):
    results = {}

    for i in range(len(config.label_to_class)):  # Class
        results[i] = {}
        # REAL:1, FAKE:0
        subset = (labels == i) & (reals == 1)
        if subset.sum() == 0:
            continue

        points = embeddings_2d[subset]
        centroid = points.mean(axis=0).tolist()
        dists = ((points - centroid) ** 2).sum(axis=1) ** 0.5
        radius = dists.mean()

        f_subset = (labels == i) & (reals == 0)
        if f_subset.sum() == 0:
            continue

        f_points = embeddings_2d[f_subset]
        f_centroid = f_points.mean(axis=0).tolist()
        f_dists = ((f_points - f_centroid) ** 2).sum(axis=1) ** 0.5
        f_radius = f_dists.mean()

        results[i] = {
            'r centroid x': centroid[0],
            'r centroid y': centroid[1],
            'r radius': radius,
            'f centroid x': f_centroid[0],
            'f centroid y': f_centroid[1],
            'f radius': f_radius,
        }

    df_dino = pd.DataFrame(results).T
    df_dino.reset_index(inplace=True)
    df_dino.rename(columns={'index': 'label'}, inplace=True)

    save_filepath = config.PROJECT_ROOT / f'results/dis_{db}_{model_name}_{reducer_name}_{seed}.csv'
    df_dino.to_csv(save_filepath, index=False)

@torch.no_grad()
def extract_embeddings(images, model_name, hf_id, batch_size):
    model_name_upper = model_name.upper()
    if model_name_upper == "CLIP":
        processor = CLIPProcessor.from_pretrained(hf_id)
        model = CLIPModel.from_pretrained(hf_id, use_safetensors=True).to(device).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(hf_id, use_fast=True)
        model = AutoModel.from_pretrained(hf_id).to(device).eval()

    embedding_batches = []

    for start in tqdm(range(0, len(images), batch_size), desc=f"Extracting {model_name} embeddings",):
        batch_images = images[start:start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt",)
        inputs = { key: value.to(device) for key, value in inputs.items()}

        if model_name_upper == "CLIP":
            features = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)

        embedding_batches.append(features.detach().cpu().numpy())

    image_embeddings = np.concatenate(embedding_batches, axis=0,)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image_embeddings


def run_reduction_pipeline(df_train, embeddings: np.ndarray, labels: np.ndarray, reals: np.ndarray, model_name: str, reducer_name: str, reducer, db, seed=42):
    embeddings_2d = reduce_dim(df_train, embeddings, reducer, model_name, reducer_name, db, seed) # <- change signature recommended
    plot_2d_embeddings(embeddings_2d, labels, reals, model_name, reducer_name, db, seed)
    save_distance(embeddings_2d, labels, reals, model_name, reducer_name, db, seed)

def _l2_normalize(x, axis=1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def save_distance_hd(embeddings_hd, labels, reals, model_name, db="FAKE1", metric="cosine", normalize=True, seed=42):
    results = []

    for i in range(len(config.label_to_class)):
        r_mask = (labels == i) & (reals == 1)
        f_mask = (labels == i) & (reals == 0)
        if r_mask.sum() == 0 or f_mask.sum() == 0:
            continue

        r = embeddings_hd[r_mask]
        f = embeddings_hd[f_mask]

        if normalize:
            r = _l2_normalize(r, axis=1)
            f = _l2_normalize(f, axis=1)

        r_cent = r.mean(axis=0)
        f_cent = f.mean(axis=0)

        if normalize:
            r_cent = r_cent / (np.linalg.norm(r_cent) + 1e-12)
            f_cent = f_cent / (np.linalg.norm(f_cent) + 1e-12)

        if metric == "cosine":
            dist = 1 - float(np.dot(r_cent, f_cent))
        elif metric == "euclidean":
            dist = float(np.linalg.norm(r_cent - f_cent))
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        results.append({"label": i, "centroid_dist_hd": dist})

    df = pd.DataFrame(results)
    save_filepath = config.PROJECT_ROOT / f"results/disHD_{db}_{model_name}_{seed}.csv"
    df.to_csv(save_filepath, index=False)

def compute_distance(db='FAKE1', seed=42):

    train_filepath = config.PROJECT_ROOT / f'train.csv'
    df = pd.read_csv(train_filepath)
    df_fake = df[df["rf"] == db]
    df_real = df[df["rf"] == "REAL"]

    ds_fake = TransDataset(df_fake)
    ds_real = TransDataset(df_real)

    images_fake, labels_fake, flags_fake = [], [], []
    for i in range(len(ds_fake)):
        img, label, flag = ds_fake[i]
        images_fake.append(img)
        labels_fake.append(label)
        flags_fake.append(0)

    labels_fake = np.array(labels_fake)
    flags_fake = np.array(flags_fake)

    images_real, labels_real, flags_real = [], [], []
    for i in range(len(ds_real)):
        img, label, flag = ds_real[i]
        images_real.append(img)
        labels_real.append(label)
        flags_real.append(1)

    labels_real = np.array(labels_real)
    flags_real = np.array(flags_real)

    images = images_fake + images_real

    labels = np.concatenate([
        np.asarray(labels_fake),
        np.asarray(labels_real),
    ])

    flags = np.concatenate([
        np.asarray(flags_fake),
        np.asarray(flags_real),
    ])

    df_selected = pd.concat(
        [df_fake, df_real],
        ignore_index=True,
    )

    for model_name, hf_id in model_specs:
        image_embeddings = extract_embeddings(images, model_name=model_name, hf_id=hf_id, batch_size=32)
        save_distance_hd(image_embeddings, labels, flags, model_name, db=db, metric="cosine", seed=seed)

        reducer_specs = [
            ("UMAP", lambda: umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)),
            ("TSNE", lambda: TSNE(n_components=2, perplexity=30, metric="cosine", random_state=seed)),
        ]

        for reducer_name, reducer_fn in reducer_specs:
            reducer = reducer_fn()
            run_reduction_pipeline(df_selected, image_embeddings, labels, flags, model_name, reducer_name, reducer, db, seed)

if __name__== "__main__":

    starttime = time.time()

    seeds = [12, 123, 1234]
    dbs = ['FAKE1', 'FAKE2']
    model_names = ["CLIP", "DINOv2", "DINOv3"]
    reducer_names = ["UMAP", "TSNE"]

    for db in dbs:
        for seed in seeds:
            compute_distance(db, seed)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))


    dfs = []

    for db in dbs:
        for model in model_names:
            for reducer in reducer_names:
                for seed in seeds:
                    fname = (
                        f"dis_{db}_{model}_{reducer}_{seed}.csv"
                    )
                    path = config.PROJECT_ROOT / "results" / fname

                    result_df = pd.read_csv(path)
                    result_df["DB"] = db
                    result_df["model"] = model
                    result_df["reducer"] = reducer
                    result_df["seed"] = seed

                    dfs.append(result_df)

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["centroid_dist"] = np.hypot(
        all_df["r centroid x"] - all_df["f centroid x"],
        all_df["r centroid y"] - all_df["f centroid y"],
    )

    all_df["relative_centroid_dist"] = (
            all_df["centroid_dist"]
            / (
                    all_df["r radius"]
                    + all_df["f radius"]
                    + 1e-9
            )
    )

    raw_path = (
            config.PROJECT_ROOT
            / "results"
            / "distance_from_centroid_all_seeds.csv"
    )
    all_df.to_csv(raw_path, index=False)

    summary_df = (
        all_df
        .groupby(
            ["DB", "model", "reducer", "label"],
            as_index=False,
        )
        .agg(
            centroid_dist_mean=("centroid_dist", "mean"),
            centroid_dist_std=("centroid_dist", "std"),
            relative_dist_mean=("relative_centroid_dist", "mean"),
            relative_dist_std=("relative_centroid_dist", "std"),
            real_radius_mean=("r radius", "mean"),
            fake_radius_mean=("f radius", "mean"),
        )
    )

    summary_path = (
            config.PROJECT_ROOT
            / "results"
            / "distance_from_centroid_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

