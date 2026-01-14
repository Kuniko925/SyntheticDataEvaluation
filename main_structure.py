from model import MobileNetV2
import config
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import data
from dataclasses import dataclass
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
os.environ['PYTHONHASHSEED'] = str(seed)

BEST_MODEL_PATH = {
    'FAKE1': {'MobileNetV2': 49, 'ResNet50': 43, 'ViT16': 46},
    'FAKE2': {'MobileNetV2': 42, 'ResNet50': 43, 'ViT16': 45},
    'REAL': {'MobileNetV2': 46, 'ResNet50': 40, 'ViT16': 47}
}

def list_mnv2_layers(model):
    names = []
    # MobileNetV2: base_model.features is Sequential
    for i in range(len(model.base_model.features)):
        names.append(f"base_model.features.{i}")
    # classifier
    names.append("base_model.classifier")
    return names


def _to_vector(act: torch.Tensor) -> torch.Tensor:
    """
    (N,C,H,W) -> (N,C) にGAP
    (N,D) ならそのまま
    """
    if act.dim() == 4:
        return act.mean(dim=(2,3))
    elif act.dim() == 2:
        return act
    else:
        return act.flatten(1)

def forward_collect_acts(model, x, layer_names):
    """
    指定した layer_names の出力を dict[name] = (N,dim) ベクトルで返す
    """
    acts = {}
    name_to_module = dict(model.named_modules())
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out が tuple の場合があるので対応
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
    """
    A,B: (N, D)
    return: cosine_mean, l2_mean, rel_l2_mean
    """
    # shape確認
    assert A.shape[0] == B.shape[0], (A.shape, B.shape)

    # cosine similarity（サンプルごと）
    A_n = F.normalize(A, dim=1)
    B_n = F.normalize(B, dim=1)
    cos = (A_n * B_n).sum(dim=1)  # (N,)

    # L2距離（サンプルごと）
    l2 = (A - B).norm(p=2, dim=1)

    # 相対L2（スケール差を吸収）
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

def load_csv_and_fix_filepath(csv_path: str, project_root) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["filepath"] = df["filepath"].str.replace(
        r'^\.\./\.\./dataset/CIFAKE/',
        str(project_root) + "/",
        regex=True
    )
    return df

def add_image_column(df: pd.DataFrame, filepath_col: str = "filepath", image_col: str = "image") -> pd.DataFrame:
    df = df.copy()
    df[image_col] = df[filepath_col].apply(os.path.basename)
    return df

def get_dataset(db):
    test_size = 0.2

    save_filepath = config.PROJECT_ROOT / f'{config.CFG[db]["DB"]}/train.csv'
    df = load_csv_and_fix_filepath(save_filepath, config.PROJECT_ROOT)
    df_real = df[df['rf'] == 'REAL'].copy()
    df_fake = df[df['rf'] == 'FAKE'].copy()
    df_train_r, df_valid_r = train_test_split(df_real, test_size=test_size, random_state=seed, shuffle=True)
    df_train_f, df_valid_f = train_test_split(df_fake, test_size=test_size, random_state=seed, shuffle=True)

    save_filepath = config.PROJECT_ROOT / f'{config.CFG[db]["DB"]}/test.csv'
    df_test = load_csv_and_fix_filepath(save_filepath, config.PROJECT_ROOT)
    df_test_r = df_test[df_test['rf'] == 'REAL'].copy()
    df_test_f = df_test[df_test['rf'] == 'FAKE'].copy()

    df_train_r = add_image_column(df_train_r)
    df_valid_r = add_image_column(df_valid_r)
    df_train_f = add_image_column(df_train_f)
    df_valid_f = add_image_column(df_valid_f)
    df_test_r = add_image_column(df_test_r)
    df_test_f = add_image_column(df_test_f)

    return df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f

def get_dataloaders(df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, batch_size=32, img_size=(32, 32)):

    label_encoder = LabelEncoder()
    label_encoder.fit(df_train_r["label"])
    train_loader_r = data.get_dataloader(df_train_r, img_size, batch_size, label_encoder, train=True)
    valid_loader_r = data.get_dataloader(df_valid_r, img_size, batch_size, label_encoder, train=False)
    test_loader_r = data.get_dataloader(df_test_r, img_size, batch_size, label_encoder, train=False)
    train_loader_f = data.get_dataloader(df_train_f, img_size, batch_size, label_encoder, train=True)
    valid_loader_f = data.get_dataloader(df_valid_f, img_size, batch_size, label_encoder, train=False)
    test_loader_f = data.get_dataloader(df_test_f, img_size, batch_size, label_encoder, train=False)

    return train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f

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
    """
    同一テストに対して
      - both_correct
      - only_A_correct
      - only_B_correct
      - both_wrong
    の割合と件数を出す。
    さらに num_class を渡すと、クラス別にも同じ集計を返す。
    """
    model_a.eval().to(device)
    model_b.eval().to(device)

    both_correct = 0
    only_a = 0
    only_b = 0
    both_wrong = 0
    total = 0

    # クラス別（任意）
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

if __name__== "__main__":

    num_class = len(config.label_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #for db, best_model in BEST_MODEL_PATH.items():
    #    for model_name, epochs in best_model.items():

    def compute_structure_difference(db):
        model_name = 'MobileNetV2'
        epochs = BEST_MODEL_PATH[db][model_name]
        best_val_file = config.PROJECT_ROOT / f'{db}/{model_name}/{db}model_{epochs}.pt'
        fake_model = MobileNetV2(num_class)
        fake_model.load_state_dict(torch.load(best_val_file, weights_only=True))

        epochs = BEST_MODEL_PATH['REAL'][model_name]
        best_val_file = config.PROJECT_ROOT / f'REAL/{model_name}/REALmodel_{epochs}.pt'
        real_model = MobileNetV2(num_class)
        real_model.load_state_dict(torch.load(best_val_file, weights_only=True))

        # to avoid randomness
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        conf = MobileNetConf()
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = get_dataset(db)
        train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = get_dataloaders(
            df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

        layer_names = list_mnv2_layers(fake_model)
        df_layer_diff = compare_models_per_layer(fake_model, real_model, test_loader_r, layer_names, device)

        save_path = config.PROJECT_ROOT / f"results/mnv2_layer_diff_{db}_vs_REAL.csv"
        df_layer_diff.to_csv(save_path, index=False)


        agree_c, support = classwise_agreement(fake_model, real_model, test_loader_r, device, num_class)
        for c, (a, n) in enumerate(zip(agree_c, support)):
            print(c, a, n)

        summary, df_cls = overlap_correctness(real_model, fake_model, test_loader_r, device, num_class=num_class)

        print(summary)
        print(df_cls.head(10))

        print("REALmodel acc on db test_real:", simple_acc(real_model, test_loader_r, device))

    DB = ['FAKE1', 'FAKE2']
    for db in DB:
        compute_structure_difference(db)





