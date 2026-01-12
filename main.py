import os
import random
import numpy as np
import pandas as pd
import data
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time
from dataclasses import dataclass


# to reproduce
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
os.environ['PYTHONHASHSEED'] = str(seed)

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


def run_training(model_class, conf, db):

    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = get_dataset(db)
    train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = get_dataloaders(
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    def preparation(model_name, db):
        # Preparation
        model_save_directory = config.PROJECT_ROOT / db / f"{config.CFG[db]['DB']}/{model_name}/{db}/" # e.g., cifake1/MobileNet/FAKE/
        utils.create_directory(model_save_directory)
        utils.delete_subfolders(model_save_directory)
        return model_save_directory

    def testing(model_name, db, test_setting, trainer, model, test_loader, df_test):
        preds = trainer.evaluate(model, test_loader)
        df_test['preds'] = preds
        df_test.to_csv(config.PROJECT_ROOT / f'results/{model_name}_{db}_{test_setting}.csv', index=False) # e.g., results/MobileNet_FAKE_REAL.csv/

    # Training Preparation
    model_save_directory = preparation(conf.model_name, db)

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader_f, valid_loader_f, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file))

    # Test
    testing(conf.model_name, db, 'REAL', trainer, model, test_loader_r, df_test_r)
    testing(conf.model_name, db, 'FAKE', trainer, model, test_loader_f, df_test_f)

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

if __name__== "__main__":

    starttime = time.time()

    DB = ['REAL', 'FAKE1', 'FAKE2']
    for db in DB:
        run_training(MobileNetV2, MobileNetConf(), db)
        run_training(ResNet50Model, ResNetConf(), db)
        run_training(ViT16, ViT16Conf(), db)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))