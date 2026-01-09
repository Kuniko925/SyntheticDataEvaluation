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


def get_dataset(root):
    random_state = 42
    test_size = 0.2

    save_filepath = f'{root}/train.csv'
    df = pd.read_csv(save_filepath)
    df['filepath'] = df['filepath'].str.replace(
        r'^\.\./\.\./dataset/CIFAKE/',
        str(config.PROJECT_ROOT) + '/',
        regex=True
    )
    df_real = df[df['rf'] == 'REAL'].copy()
    df_fake = df[df['rf'] == 'FAKE'].copy()

    df_train_r, df_valid_r = train_test_split(df_real, test_size=test_size, random_state=random_state, shuffle=True)
    df_train_r['image'] = df_train_r['filepath'].apply(os.path.basename)
    df_valid_r['image'] = df_valid_r['filepath'].apply(os.path.basename)

    df_train_f, df_valid_f = train_test_split(df_fake, test_size=test_size, random_state=random_state, shuffle=True)
    df_train_f['image'] = df_train_f['filepath'].apply(os.path.basename)
    df_valid_f['image'] = df_valid_f['filepath'].apply(os.path.basename)

    save_filepath = f'{root}/test.csv'
    df_test = pd.read_csv(save_filepath)
    df_test['filepath'] = df_test['filepath'].str.replace(
        r'^\.\./\.\./dataset/CIFAKE/',
        str(config.PROJECT_ROOT) + '/',
        regex=True
    )
    df_test_r = df_test[df_test['rf'] == 'REAL'].copy()
    df_test_f = df_test[df_test['rf'] == 'FAKE'].copy()
    df_test_r['image'] = df_test_r['filepath'].apply(os.path.basename)
    df_test_f['image'] = df_test_f['filepath'].apply(os.path.basename)

    return df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f

def get_dataloaders(df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, batch_size=32, img_size=(32, 32)):

    num_class = len(config.label_to_class)
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train_r["label"])
    train_loader_r = data.get_dataloader(df_train_r, img_size, batch_size, label_encoder, train=True)
    valid_loader_r = data.get_dataloader(df_valid_r, img_size, batch_size, label_encoder, train=False)
    test_loader_r = data.get_dataloader(df_test_r, img_size, batch_size, label_encoder, train=False)
    train_loader_f = data.get_dataloader(df_train_f, img_size, batch_size, label_encoder, train=True, brightness=0.4, contrast=0.4)
    valid_loader_f = data.get_dataloader(df_valid_f, img_size, batch_size, label_encoder, train=False)
    test_loader_f = data.get_dataloader(df_test_f, img_size, batch_size, label_encoder, train=False)

    return train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f


def run_training(model_class, **kwargs):
    num_class = kwargs.get('num_class')
    num_epochs = kwargs.get('epochs')
    lr = kwargs.get('lr')
    model_name = kwargs.get('model_name')
    root = kwargs.get('root')
    batch_size = kwargs.get('batch_size')
    img_size = kwargs.get('img_size')

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = get_dataset(root)
    train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = get_dataloaders(
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, batch_size, img_size)

    def preparation(model_save_directory):
        # Preparation
        utils.create_directory(model_save_directory)
        utils.delete_subfolders(model_save_directory)

    def real_testing(root, model_name, setting, trainer, model, test_loader, df_test):
        display = True
        out_filepath = config.PROJECT_ROOT / f'results/{model_name}_{setting}_real_report.csv'

        preds = trainer.evaluate(model, test_loader, display, out_filepath)
        df_test['preds'] = preds
        df_test.to_csv(config.PROJECT_ROOT / f'results/{model_name}_{setting}_real.csv', index=False)

    def fake_testing(root, model_name, setting, trainer, model, test_loader, df_test):
        display = True
        out_filepath = config.PROJECT_ROOT / f'results/{model_name}_{setting}_fake_report.csv'

        preds = trainer.evaluate(model, test_loader, display, out_filepath)
        df_test['preds'] = preds
        df_test.to_csv(config.PROJECT_ROOT / f'results/{model_name}_{setting}_fake.csv', index=False)


    # Fake Setting Preparation
    setting = 'fake1_agu'
    model_save_directory = root / f"{model_name}/{setting}/"
    preparation(model_save_directory)

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader_f, valid_loader_f, model_save_directory, epochs=num_epochs, lr=lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file))

    # Real
    real_testing(root, model_name, setting, trainer, model, test_loader_r, df_test_r)
    # Fake
    fake_testing(root, model_name, setting, trainer, model, test_loader_f, df_test_f)


if __name__== "__main__":


    root = config.PROJECT_ROOT / "cifake1"
    num_class = len(config.label_to_class)
    model_name = config.MODELS[1]
    num_epochs = 100
    lr = 1e-2
    batch_size = 32
    img_size = (32, 32)

    settings = {
        'num_class': num_class, 'epochs': num_epochs, 'lr':lr, 'model_name': model_name, 'root': root, 'batch_size': batch_size, 'img_size': img_size}
    run_training(MobileNetV2, **settings)

    model_name = config.MODELS[0]
    num_epochs = 100
    lr = 1e-2

    settings = {
        'num_class': num_class, 'epochs': num_epochs, 'lr': lr, 'model_name': model_name, 'root': root,
        'batch_size': batch_size, 'img_size': img_size}
    run_training(ResNet50Model, **settings)

    model_name = 'ViT16'
    num_epochs = 100
    lr = 1e-5
    batch_size = 64
    img_size = (224, 224)

    settings = {
        'num_class': num_class, 'epochs': num_epochs, 'lr': lr, 'model_name': model_name, 'root': root,
        'batch_size': batch_size, 'img_size': img_size}
    run_training(ViT16, **settings)