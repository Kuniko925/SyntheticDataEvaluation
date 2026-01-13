import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap
from data import TransDataset
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPModel
import config

def reduce_dim(df_train, image_embeddings, reducer, model_name, reducer_name, db='FAKE1'):
    embeddings_2d = reducer.fit_transform(image_embeddings)
    df_train['embeddings x'] = embeddings_2d[:,0]
    df_train['embeddings y'] = embeddings_2d[:,1]
    save_filepath = config.PROJECT_ROOT / f'results/embed_{db}_{model_name}_{reducer_name}.csv'
    df_train.to_csv(save_filepath, index=False)
    return embeddings_2d


def plot_2d_embeddings(embeddings_2d, labels, reals, model_name, reducer_name, db='FAKE1'):
    title = f"{model_name} Embeddings per Class by {reducer_name}"
    save_filepath = config.PROJECT_ROOT / f'results/{db}_{model_name}_{reducer_name}.png'

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

        ax.set_title(f"{config.label_to_class[i]}")
        ax.grid(True)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_filepath)
    plt.close(fig)


def save_distance(embeddings_2d, labels, reals, model_name, reducer_name, db='FAKE1'):
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

    save_filepath = config.PROJECT_ROOT / f'results/dis_{db}_{model_name}_{reducer_name}.csv'
    df_dino.to_csv(save_filepath, index=False)

@torch.no_grad()
def extract_embeddings(images, model_name, hf_id, batch_size, num_workers):
    model_name_upper = model_name.upper()
    if model_name_upper == "CLIP":
        processor = CLIPProcessor.from_pretrained(hf_id)
        model = CLIPModel.from_pretrained(hf_id, use_safetensors=True).to(device).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(hf_id, use_fast=True)
        model = AutoModel.from_pretrained(hf_id).to(device).eval()

    image_embeddings = []
    for img in tqdm(images):
        inputs = processor(images=img, return_tensors="pt").to(device)
        if model_name_upper == "CLIP":
            feats = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
            feats = outputs.last_hidden_state.mean(dim=1)
        image_embeddings.append(feats.cpu().numpy().flatten())

    return np.array(image_embeddings)


def run_reduction_pipeline(
        df_train,
        embeddings: np.ndarray,
        labels: np.ndarray,
        reals: np.ndarray,
        model_name: str,
        reducer_name: str,
        reducer,
        db
):
    embeddings_2d = reduce_dim(df_train, embeddings, reducer, model_name, reducer_name, db) # <- change signature recommended
    plot_2d_embeddings(embeddings_2d, labels, reals, model_name, reducer_name, db)
    save_distance(embeddings_2d, labels, reals, model_name, reducer_name, db)

if __name__== "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model_specs = [
        ("DINOv2", "facebook/dinov2-small"),
        ("DINOv3", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        ("CLIP", "openai/clip-vit-base-patch32"),
    ]

    reducer_specs = [
        ("UMAP", lambda: umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")),
        ("TSNE", lambda: TSNE(n_components=2, perplexity=30, metric="cosine", random_state=42)),
    ]

    def compute_distance(train_filepath, db='FAKE1'):
        df_train = pd.read_csv(train_filepath)
        df_train["filepath"] = df_train["filepath"].str.replace(
            r'^\.\./\.\./dataset/CIFAKE/',
            str(config.PROJECT_ROOT) + "/",
            regex=True
        )

        class_labels = list(config.label_to_class.values())
        dataset = TransDataset(df_train)

        images, labels, reals = [], [], []
        for i in range(len(dataset)):
            img, label, real = dataset[i]
            images.append(img)
            labels.append(label)
            reals.append(real)

        labels = np.array(labels)
        reals = np.array(reals)

        for model_name, hf_id in model_specs:
            image_embeddings = extract_embeddings(
                images,
                model_name=model_name,
                hf_id=hf_id,
                batch_size=32,
                num_workers=2,
            )

            for reducer_name, reducer_fn in reducer_specs:
                reducer = reducer_fn()
                run_reduction_pipeline(df_train, image_embeddings, labels, reals, model_name, reducer_name, reducer, db)


    train_filepath = config.PROJECT_ROOT / f'cifake1/train.csv'
    db = 'FAKE1'
    compute_distance(train_filepath, db)
    train_filepath = config.PROJECT_ROOT / f'cifake2/train.csv'
    db = 'FAKE2'
    compute_distance(train_filepath, db)