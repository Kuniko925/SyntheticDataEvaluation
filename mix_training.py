import os
import random
import numpy as np
import data
import utils
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time
import train_settings
import pandas as pd


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


MODEL_REGISTRY = {
  "MobileNetV2": {
      "ctor": MobileNetV2,
      "conf": train_settings.MobileNetConf(),
  },
  "ResNet50": {
      "ctor": ResNet50Model,
      "conf": train_settings.ResNetConf(),
  },
  "ViT16": {
      "ctor": ViT16,
      "conf": train_settings.ViT16Conf(),
  },
}

def stratified_sample(df, label_col="label", frac=0.20, random_state=seed):
    return df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)

def preparation(model_name, real_raito):
    # Preparation
    model_save_directory = config.PROJECT_ROOT / f"MIX/{model_name}/{str(int(real_raito*100))}/" # e.g., layerwise/MobileNet/10/
    utils.create_directory(model_save_directory)
    utils.delete_subfolders(model_save_directory)
    return model_save_directory

def mix_finetune(model_name, real_ratio=0.1):

    model_class = MODEL_REGISTRY[model_name]["ctor"]
    conf = MODEL_REGISTRY[model_name]["conf"]
    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=real_ratio)
    df_train_f_sub = stratified_sample(df_train_f, label_col="label", frac=(1-real_ratio))
    df_valid_r_sub = stratified_sample(df_valid_r, label_col="label", frac=real_ratio)
    df_valid_f_sub = stratified_sample(df_valid_f, label_col="label", frac=(1-real_ratio))
    df_train = pd.concat([df_train_r_sub, df_train_f_sub], axis=0)
    df_valid = pd.concat([df_valid_r_sub, df_valid_f_sub], axis=0)

    train_loader, valid_loader, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = data.get_dataloaders(
        df_train, df_valid, df_test_r, df_train, df_valid, df_test_f, conf.batch_size, conf.img_size)

    # Training Preparation
    model_save_directory = preparation(model_name, real_ratio)

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file, weights_only=False))

    # Test
    df_test_r['preds'] = trainer.evaluate(model, test_loader_r)
    df_test_r.to_csv(config.PROJECT_ROOT / f'results/{model_name}_mix_{str(int(real_ratio * 100))}.csv', index=False)

if __name__== "__main__":

    starttime = time.time()

    real_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_names = config.MODELS
    for model_name in model_names:
        for real_ratio in real_ratios:
            mix_finetune(model_name, real_ratio=real_ratio)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))