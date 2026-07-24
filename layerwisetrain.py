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

def set_layerwise_modeling(model, freeze_until=9, freeze_bn=True):

    for i, block in enumerate(model.base_model.features):
        req_grad = (i < freeze_until)
        for p in block.parameters():
            p.requires_grad = req_grad

        if freeze_bn and (i >= freeze_until):
            block.eval()

    for p in model.base_model.classifier.parameters():
        p.requires_grad = True

MODEL_REGISTRY = {
  "MobileNetV2": {
      "ctor": MobileNetV2,
      "conf": train_settings.MobileNetConf(),
      "FAKE1_best": "FAKE1/MobileNetV2/FAKE1model_49.pt",
      'freeze': 3,
  },
  "ResNet50": {
      "ctor": ResNet50Model,
      "conf": train_settings.ResNetConf(),
      "FAKE1_best": "FAKE1/ResNet50Model/FAKE1model_43.pt",
      'freeze': 8,
  },
  "ViT16": {
      "ctor": ViT16,
      "conf": train_settings.ViT16Conf(),
      "FAKE1_best": "FAKE1/ViT16/FAKE1model_46.pt",
      'freeze': 5,
  },
}


def stratified_sample(df, label_col="label", frac=0.10, random_state=seed):
    return df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)

def preparation(model_name, real_raito):
    # Preparation
    model_save_directory = config.PROJECT_ROOT / f"Layerwise/{model_name}/" # e.g., layerwise/MobileNet/10/
    utils.create_directory(model_save_directory)
    utils.delete_subfolders(model_save_directory)
    return model_save_directory

def finetune_layerwise(model_name, real_raito=0.1):

    model_class = MODEL_REGISTRY[model_name]["ctor"]
    conf = MODEL_REGISTRY[model_name]["conf"]
    fake1_best_model = MODEL_REGISTRY[model_name]["FAKE1_best"]
    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=real_raito)
    df_valid_r_sub = stratified_sample(df_valid_r, label_col="label", frac=real_raito)

    train_loader, valid_loader, test_loader_r, _, _, _ = data.get_dataloaders(
        df_train_r_sub, df_valid_r_sub, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    # Training Preparation
    model_save_directory = preparation(model_name, real_raito)

    # Training
    model = model_class(num_class)
    best_val_file = config.PROJECT_ROOT / fake1_best_model
    model.load_state_dict(torch.load(best_val_file))
    set_layerwise_modeling(model, freeze_until=3, freeze_bn=True)

    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file, weights_only=False))

    # Test
    df_test_r['preds'] = trainer.evaluate(model, test_loader_r)
    df_test_r.to_csv(config.PROJECT_ROOT / f'results/{model_name}_layerwise.csv', index=False)


if __name__== "__main__":

    starttime = time.time()

    model_names = config.MODELS
    for model_name in model_names:
        finetune_layerwise(model_name, real_raito=0.1)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))