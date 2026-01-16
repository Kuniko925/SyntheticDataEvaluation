from model import MobileNetV2, ResNet50Model, ViT16
import config
import torch
import torch.nn.functional as F
import pandas as pd
import os
import data
from dataclasses import dataclass
import random
import numpy as np
import re
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
os.environ['PYTHONHASHSEED'] = str(seed)



def list_mnv2_layers(model):
    names = []
    # MobileNetV2: base_model.features is Sequential
    for i in range(len(model.base_model.features)):
        names.append(f"base_model.features.{i}")
    # classifier
    names.append("base_model.classifier")
    return names

def list_resnet50_layers(model):
    names = []
    names += [
        "base_model.conv1",
        "base_model.bn1",
        "base_model.relu",
        "base_model.maxpool",
    ]
    for s in range(1, 5):  # layer1..layer4
        layer = getattr(model.base_model, f"layer{s}")
        for b in range(len(layer)):
            names.append(f"base_model.layer{s}.{b}")
    if hasattr(model.base_model, "avgpool"):
        names.append("base_model.avgpool")
    if hasattr(model.base_model, "fc"):
        names.append("base_model.fc")
    else:
        names.append("base_model.classifier")

    return names


def _to_vector(act: torch.Tensor) -> torch.Tensor:
    if act.dim() == 4:
        return act.mean(dim=(2,3))
    elif act.dim() == 2:
        return act
    else:
        return act.flatten(1)

def forward_collect_acts(model, x, layer_names):
    acts = {}
    name_to_module = dict(model.named_modules())
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out_ = out[0]
            else:
                out_ = out
            acts[name] = _to_vector(out_.detach())
        return hook

    for name in layer_names:
        if name not in name_to_module:
            raise KeyError(f"Layer '{name}' not found. Available example: {list(name_to_module.keys())[:20]}")
        hooks.append(name_to_module[name].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return acts

def compare_layer(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[0] == B.shape[0], (A.shape, B.shape)

    # Cosine similarity
    A_n = F.normalize(A, dim=1)
    B_n = F.normalize(B, dim=1)
    cos = (A_n * B_n).sum(dim=1)  # (N,)

    # L2 distance
    l2 = (A - B).norm(p=2, dim=1)

    # Relative L2 distance
    denom = A.norm(p=2, dim=1).clamp_min(1e-8)
    rel = l2 / denom

    return cos.mean().item(), l2.mean().item(), rel.mean().item()


def compare_models_per_layer(fake_model, real_model, test_loader, layer_names, device):

    sums = {name: {"cos": 0.0, "l2": 0.0, "rel": 0.0, "n": 0} for name in layer_names}

    fake_model = fake_model.to(device).eval()
    real_model = real_model.to(device).eval()

    for images, labels, _ in test_loader:
        images = images.to(device, non_blocking=True)

        acts_fake = forward_collect_acts(fake_model, images, layer_names)
        acts_real = forward_collect_acts(real_model, images, layer_names)

        bs = images.size(0)
        for name in layer_names:
            A = acts_fake[name]
            B = acts_real[name]
            cos_m, l2_m, rel_m = compare_layer(A, B)

            sums[name]["cos"] += cos_m * bs
            sums[name]["l2"]  += l2_m  * bs
            sums[name]["rel"] += rel_m * bs
            sums[name]["n"]   += bs

    rows = []
    for name in layer_names:
        n = sums[name]["n"]
        rows.append({
            "layer": name,
            "cosine_mean": sums[name]["cos"] / n,
            "l2_mean":     sums[name]["l2"]  / n,
            "rel_l2_mean": sums[name]["rel"] / n,
        })

    df = pd.DataFrame(rows)
    return df

@dataclass
class TrainConf:
    model_name: str
    lr: float
    batch_size: int
    img_size: tuple[int, int]
    num_epochs: int

def ResNetConf():
    return TrainConf("ResNet50", 1e-2, 32, (32, 32), 50)

def MobileNetConf():
    return TrainConf("MobileNetV2", 1e-2, 32, (32, 32), 50)

def ViT16Conf():
    return TrainConf("ViT16", 1e-5, 64, (224, 224), 50)


def classwise_agreement(model_a, model_b, test_loader, device, num_class):
    model_a.eval().to(device)
    model_b.eval().to(device)

    same = np.zeros(num_class, dtype=np.int64)
    cnt = np.zeros(num_class, dtype=np.int64)

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            pred_a = model_a(x).argmax(dim=1)
            pred_b = model_b(x).argmax(dim=1)

            for c in range(num_class):
                mask = (y == c)
                cnt[c] += mask.sum().item()
                same[c] += ((pred_a == pred_b) & mask).sum().item()

    return same / np.clip(cnt, 1, None), cnt

def overlap_correctness(model_a, model_b, test_loader, device, num_class=None):

    model_a.eval().to(device)
    model_b.eval().to(device)

    both_correct = 0
    only_a = 0
    only_b = 0
    both_wrong = 0
    total = 0

    if num_class is not None:
        cls_stats = {c: {"both_correct":0, "only_a":0, "only_b":0, "both_wrong":0, "total":0} for c in range(num_class)}
    else:
        cls_stats = None

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            pa = model_a(x).argmax(dim=1)
            pb = model_b(x).argmax(dim=1)

            ca = (pa == y)  # A correct?
            cb = (pb == y)  # B correct?

            bc = (ca & cb)
            oa = (ca & ~cb)
            ob = (~ca & cb)
            bw = (~ca & ~cb)

            bs = y.size(0)
            both_correct += bc.sum().item()
            only_a      += oa.sum().item()
            only_b      += ob.sum().item()
            both_wrong  += bw.sum().item()
            total       += bs

            if cls_stats is not None:
                for c in range(num_class):
                    mask = (y == c)
                    n = mask.sum().item()
                    if n == 0:
                        continue
                    cls_stats[c]["both_correct"] += (bc & mask).sum().item()
                    cls_stats[c]["only_a"]       += (oa & mask).sum().item()
                    cls_stats[c]["only_b"]       += (ob & mask).sum().item()
                    cls_stats[c]["both_wrong"]   += (bw & mask).sum().item()
                    cls_stats[c]["total"]        += n

    summary = {
        "both_correct": both_correct,
        "only_A_correct": only_a,
        "only_B_correct": only_b,
        "both_wrong": both_wrong,
        "total": total,
        "both_correct_rate": both_correct / total,
        "only_A_correct_rate": only_a / total,
        "only_B_correct_rate": only_b / total,
        "both_wrong_rate": both_wrong / total,
    }

    df_cls = None
    if cls_stats is not None:
        rows = []
        for c, d in cls_stats.items():
            t = d["total"]
            if t == 0:
                continue
            rows.append({
                "class": c,
                **d,
                "both_correct_rate": d["both_correct"]/t,
                "only_A_correct_rate": d["only_a"]/t,
                "only_B_correct_rate": d["only_b"]/t,
                "both_wrong_rate": d["both_wrong"]/t,
            })
        df_cls = pd.DataFrame(rows).sort_values("both_correct_rate")

    return summary, df_cls

def simple_acc(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

def layer_index_mnv2(name: str) -> int:
    m = re.search(r"base_model\.features\.(\d+)$", str(name))
    if m:
        return int(m.group(1))
    if str(name).endswith("base_model.classifier") or str(name).endswith("classifier"):
        return 10**9
    return 10**8

def layer_index_resnet(name: str) -> int:
    name = str(name)

    # stem: 0-3
    stem = {
        "base_model.conv1": 0,
        "base_model.bn1": 1,
        "base_model.relu": 2,
        "base_model.maxpool": 3,
    }
    if name in stem:
        return stem[name]
    m = re.search(r"base_model\.layer([1-4])\.(\d+)$", name)
    if m:
        s = int(m.group(1))
        b = int(m.group(2))
        return s * 100 + b

    # tail
    if name == "base_model.avgpool":
        return 900
    if name in ("base_model.fc", "base_model.classifier"):
        return 901
    return 999


def list_vit_layers(model):
    names = ["base_model.conv_proj"]  # patch embed

    # encoder blocks: base_model.encoder.layers.encoder_layer_0
    name_to_module = dict(model.named_modules())
    idxs = []
    for k in name_to_module.keys():
        m = re.match(r"base_model\.encoder\.layers\.encoder_layer_(\d+)$", k)
        if m:
            idxs.append(int(m.group(1)))

    if not idxs:
        raise KeyError("No ViT encoder layers found (expected base_model.encoder.layers.encoder_layer_i).")

    for i in sorted(idxs):
        names.append(f"base_model.encoder.layers.encoder_layer_{i}")

    # final norm + head
    if "base_model.encoder.ln" in name_to_module:
        names.append("base_model.encoder.ln")
    names.append("base_model.heads")

    return names


def layer_index_vit(name: str) -> int:
    name = str(name)
    if name == "base_model.conv_proj":
        return 0

    m = re.match(r"base_model\.encoder\.layers\.encoder_layer_(\d+)$", name)
    if m:
        return 100 + int(m.group(1))

    if name == "base_model.encoder.ln":
        return 900
    if name == "base_model.heads":
        return 901
    return 999


MODEL_REGISTRY = {
  "MobileNetV2": {
      "ctor": MobileNetV2,
      "conf": MobileNetConf(),
      "list_layers": list_mnv2_layers,
      "layer_index": layer_index_mnv2,
  },
  "ResNet50": {
      "ctor": ResNet50Model,
      "conf": ResNetConf(),
      "list_layers": list_resnet50_layers,
      "layer_index": layer_index_resnet,
  },
  "ViT16": {
      "ctor": ViT16,
      "conf": ViT16Conf(),
      "list_layers": list_vit_layers,
      "layer_index": layer_index_vit,
  },
}


def add_layer_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def layer_to_index(name: str) -> int:
        m = re.search(r"features\.(\d+)$", str(name))
        if m:
            return int(m.group(1))
        if str(name).endswith("classifier"):
            return 10 ** 9
        return 10 ** 8

    df["layer_index"] = df["layer"].apply(layer_to_index)

    feat_mask = df["layer"].astype(str).str.contains(r"features\.\d+$", regex=True)
    if feat_mask.any():
        max_feat = int(df.loc[feat_mask, "layer_index"].max())
        df.loc[df["layer"].astype(str).str.endswith("classifier"), "layer_index"] = max_feat + 1

    return df.sort_values("layer_index").reset_index(drop=True)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_layer_index(df)

def compute_structure_difference(model_name, db):

    epochs = config.BEST_MODEL_PATH[db][model_name]
    best_val_file = config.PROJECT_ROOT / f'{db}/{model_name}/{db}model_{epochs}.pt'

    if model_name == "ResNet50":
        fake_model = ResNet50Model(num_class)
    elif model_name == "MobileNetV2":
        fake_model = MobileNetV2(num_class)
    else:
        fake_model = ViT16(num_class)
    fake_model.load_state_dict(torch.load(best_val_file, weights_only=True))

    epochs = config.BEST_MODEL_PATH['REAL'][model_name]
    best_val_file = config.PROJECT_ROOT / f'REAL/{model_name}/REALmodel_{epochs}.pt'

    if model_name == "ResNet50":
        real_model = ResNet50Model(num_class)
    elif model_name == "MobileNetV2":
        real_model = MobileNetV2(num_class)
    else:
        real_model = ViT16(num_class)
    real_model.load_state_dict(torch.load(best_val_file, weights_only=True))

    # to avoid randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_name == "MobileNetV2":
        conf = MobileNetConf()
    elif model_name == "ResNet50":
        conf = ResNetConf()
    elif model_name == "ViT16":
        conf = ViT16Conf()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset(db)
    train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = data.get_dataloaders(
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    if model_name == "MobileNetV2":
        layer_names = list_mnv2_layers(fake_model)
    elif model_name == "ResNet50":
        layer_names = list_resnet50_layers(fake_model)
    elif model_name == "ViT16":
        layer_names = list_vit_layers(fake_model)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    df_layer_diff = compare_models_per_layer(fake_model, real_model, test_loader_r, layer_names, device)

    save_path = config.PROJECT_ROOT / f"results/{model_name}_layer_diff_{db}_vs_REAL.csv"
    df_layer_diff.to_csv(save_path, index=False)


    agree_c, support = classwise_agreement(fake_model, real_model, test_loader_r, device, num_class)
    for c, (a, n) in enumerate(zip(agree_c, support)):
        print(c, a, n)

    summary, df_cls = overlap_correctness(real_model, fake_model, test_loader_r, device, num_class=num_class)

    print(summary)
    print(df_cls.head(10))

    print("REAL model acc on db test_real:", simple_acc(real_model, test_loader_r, device))


def plot_layer_difference(model_name):
    FAKE1_CSV = f"results/{model_name}_layer_diff_FAKE1_vs_REAL.csv"
    FAKE2_CSV = f"results/{model_name}_layer_diff_FAKE2_vs_REAL.csv"

    df1 = load(FAKE1_CSV)
    df2 = load(FAKE2_CSV)

    plt.figure()
    sc1 = plt.scatter(df1["rel_l2_mean"], df1["cosine_mean"], c=df1["layer_index"],
                      marker="o", alpha=0.9, label="FAKE1")
    sc2 = plt.scatter(df2["rel_l2_mean"], df2["cosine_mean"], c=df2["layer_index"],
                      marker="^", alpha=0.9, label="FAKE2")

    plt.xlabel("l2_mean (lower = closer)")
    plt.ylabel("cosine_mean (higher = closer)")
    plt.title(f"{model_name} Layer-wise similarity on REAL")
    plt.legend()

    cbar = plt.colorbar(sc2)
    cbar.set_label("layer index (depth)")
    save_filepath = config.PROJECT_ROOT / f"results/{model_name}_layer_diff_FAKE1_vs_REAL.png"
    plt.savefig(save_filepath)
    plt.close()

if __name__== "__main__":

    num_class = len(config.label_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DB = ['FAKE1', 'FAKE2']
    model_names = config.MODELS
    for db in DB:
        for model_name in model_names:
            compute_structure_difference(model_name, db)
            plot_layer_difference(model_name)





