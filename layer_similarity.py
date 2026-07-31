from model import MobileNetV2, ResNet50Model, ViT16
import config
import torch.nn.functional as F
import data
import re
import matplotlib.pyplot as plt
import seaborn as sns
from train_utils import *
from train_settings import *

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

            ca = (pa == y)  # A correct
            cb = (pb == y)  # B correct

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

def list_mnv2_layers(model):
    names = []
    for i in range(len(model.base_model.features)):
        names.append(f"base_model.features.{i}")
    return names

def layer_index_mnv2(name: str) -> int:
    m = re.search(r"base_model\.features\.(\d+)$", str(name))
    if m:
        return int(m.group(1))
    return 999

def list_resnet50_layers(model):
    names = ["base_model.conv1"]
    for s in range(1, 5):  # layer1..layer4
        layer = getattr(model.base_model, f"layer{s}")
        for b in range(len(layer)):
            names.append(f"base_model.layer{s}.{b}")
    return names


def layer_index_resnet(name: str) -> int:
    name = str(name)
    stem = {
        "base_model.conv1": 0
    }
    if name in stem:
        return stem[name]
    m = re.search(r"base_model\.layer([1-4])\.(\d+)$", name)
    if m:
        s = int(m.group(1))
        b = int(m.group(2))
        return s * 100 + b
    return 999


def list_vit_layers(model):
    names = []

    # encoder blocks: base_model.encoder.layers.encoder_layer_0
    name_to_module = dict(model.named_modules())
    idxs = []
    for k in name_to_module.keys():
        m = re.match(r"base_model\.encoder\.layers\.encoder_layer_(\d+)$", k)
        if m:
            idxs.append(int(m.group(1)))

    for i in sorted(idxs):
        names.append(f"base_model.encoder.layers.encoder_layer_{i}")
    return names


def layer_index_vit(name: str) -> int:
    name = str(name)
    m = re.match(r"base_model\.encoder\.layers\.encoder_layer_(\d+)$", name)
    if m:
        return int(m.group(1))
    return 999 # for other name


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

def compute_structure_difference(model_name, db):

    seed = 123
    spec = MODEL_REGISTRY[model_name]
    model_class = spec["ctor"]
    conf = spec["conf"]
    list_layers = spec["list_layers"]

    best_val_file = config.PROJECT_ROOT / f'{db}/{model_name}_{seed}_best.pt'
    fake_model = model_class(num_class)
    fake_model.load_state_dict(torch.load(best_val_file, weights_only=True))

    best_val_file = config.PROJECT_ROOT / f'REAL/{model_name}_{seed}_best.pt'
    real_model = model_class(num_class)
    real_model.load_state_dict(torch.load(best_val_file, weights_only=True))

    # to avoid randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df_test_real = data.get_test_data('REAL')
    test_loader_real = data.get_test_loader(df_test_real, conf.batch_size, conf.img_size)

    layer_names = list_layers(fake_model)
    df_layer_diff = compare_models_per_layer(fake_model, real_model, test_loader_real, layer_names, device)

    save_path = config.PROJECT_ROOT / f"results/{model_name}_layer_diff_{db}_vs_REAL_{seed}.csv"
    df_layer_diff.to_csv(save_path, index=False)

    agree_c, support = classwise_agreement(fake_model, real_model, test_loader_real, device, num_class)
    df_agree = pd.DataFrame({
        "class": np.arange(num_class),
        "agreement": agree_c,
        "support": support,
    })

    df_agree["class_name"] = df_agree["class"].map(config.label_to_class)
    df_agree["db"] = db
    df_agree["model"] = model_name
    df_agree["A"] = db
    df_agree["B"] = "REAL"

    out_path = config.PROJECT_ROOT / f"results/{model_name}_agreement_{db}_vs_REAL.csv"
    df_agree.to_csv(out_path, index=False)
    print("saved:", out_path)

    print("REAL model acc on db test_real:", simple_acc(real_model, test_loader_real, device))
    summary, df_cls = overlap_correctness(real_model, fake_model, test_loader_real, device, num_class=num_class)
    # summary
    summary_path = config.PROJECT_ROOT / f"results/{model_name}_overlap_summary_{db}_vs_REAL.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    # class
    if df_cls is not None:
        cls_path = config.PROJECT_ROOT / f"results/{model_name}_overlap_byclass_{db}_vs_REAL.csv"
        df_cls.to_csv(cls_path, index=False)

    df_dist = compare_models_per_layer_dist(fake_model, real_model, test_loader_real, layer_names, device)
    df_dist["db"] = db
    df_dist["model"] = model_name
    df_dist["layer_short"] = df_dist["layer"].apply(lambda x: short_layer_name(model_name, x))

    return summary, df_cls, df_dist

def short_layer_name(model_name: str, s: str) -> str:
    s = str(s).replace("base_model.", "")

    if model_name == "ResNet50":
        s = s.replace("layer", "L")
        return s

    if model_name == "ViT16":
        s = s.replace("encoder.layers.encoder_layer", "L")
        return s

    if model_name == "MobileNetV2":
        s = s.replace("features.", "L")
        return s

    return s


def compare_models_per_layer_dist(fake_model, real_model, test_loader, layer_names, device):
    fake_model = fake_model.to(device).eval()
    real_model = real_model.to(device).eval()

    rows = []

    for images, labels, _ in test_loader:
        images = images.to(device, non_blocking=True)

        acts_fake = forward_collect_acts(fake_model, images, layer_names)
        acts_real = forward_collect_acts(real_model, images, layer_names)

        bs = images.size(0)

        for name in layer_names:
            A = acts_fake[name]  # (N, D)
            B = acts_real[name]

            A_n = F.normalize(A, dim=1)
            B_n = F.normalize(B, dim=1)
            cos = (A_n * B_n).sum(dim=1)  # (N,)

            l2 = (A - B).norm(p=2, dim=1)
            denom = A.norm(p=2, dim=1).clamp_min(1e-8)
            rel = l2 / denom

            rows.append(pd.DataFrame({
                "layer": [name] * bs,
                "cosine": cos.detach().cpu().numpy(),
                "l2": l2.detach().cpu().numpy(),
                "rel_l2": rel.detach().cpu().numpy(),
            }))

    return pd.concat(rows, ignore_index=True)

def plot_layer_dist_box(df_long, value_col, out_path):
    legend_maps = {'FAKE1':"SDGen", "FAKE2": "EDMGen"}
    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(data=df_long, x="layer_short", y=value_col, hue="db", showfliers=False)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [legend_maps.get(l, l) for l in labels]
    ax.legend(handles, new_labels, title="")

    ax.set_ylabel(value_col, fontsize=14)
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    ax.tick_params(axis="x", labelsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__== "__main__":

    num_class = len(config.label_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dfs = []

    DB = ['FAKE1', 'FAKE2']
    model_names = config.MODELS
    for model_name in model_names:
        dfs_dist = []
        for db in DB:
            summary, df_cls, df_dist = compute_structure_difference(model_name, db)
            dfs.append(df_cls)
            dfs_dist.append(df_dist)
            df_all = pd.concat(dfs_dist, ignore_index=True)

            plot_layer_dist_box(
                df_all,
                value_col="cosine",
                out_path=config.PROJECT_ROOT / f"results/{model_name}_cosine_box.png",
            )

            plot_layer_dist_box(
                df_all,
                value_col="l2",
                out_path=config.PROJECT_ROOT / f"results/{model_name}_L2_box.png",
            )





