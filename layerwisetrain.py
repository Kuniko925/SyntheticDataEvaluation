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

def set_layerwise_modeling(model, freeze_until=9, freeze_bn=True):

    for i, block in enumerate(model.base_model.features):
        req_grad = (i < freeze_until)
        for p in block.parameters():
            p.requires_grad = req_grad

        if freeze_bn and (i >= freeze_until):
            block.eval()

    for p in model.base_model.classifier.parameters():
        p.requires_grad = True

def finetune_layerwise(model_class, conf, real_raito=0.1):

    num_class = len(config.label_to_class)

    def stratified_sample(df, label_col="label", frac=0.20, random_state=42):
        return (df.groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(frac=frac, random_state=random_state))
                .reset_index(drop=True))

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=0.10, random_state=42)
    # df_valid_r_sub = stratified_sample(df_valid_r, label_col="label", frac=0.10, random_state=42)

    train_loader_r_sub, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = data.get_dataloaders(
        df_train_r_sub, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    def preparation(model_name, real_raito):
        # Preparation
        model_save_directory = config.PROJECT_ROOT / f"Layerwise/{model_name}/{str(real_raito*10)}/" # e.g., layerwise/MobileNet/10/
        utils.create_directory(model_save_directory)
        utils.delete_subfolders(model_save_directory)
        return model_save_directory

    def testing(model_name, test_setting, trainer, model, test_loader, df_test):
        preds = trainer.evaluate(model, test_loader)
        df_test['preds'] = preds
        df_test.to_csv(config.PROJECT_ROOT / f'results/{model_name}_layerwise_{str(real_raito*10)}_{test_setting}.csv', index=False) # e.g., results/MobileNet_FAKE_REAL.csv/

    # Training Preparation
    model_save_directory = preparation(conf.model_name, real_raito)

    # Training
    model = model_class(num_class)
    best_val_file = config.PROJECT_ROOT / f'FAKE1/{model_name}/FAKE1model_49.pt'
    model.load_state_dict(torch.load(best_val_file))
    set_layerwise_modeling(model, freeze_until=3, freeze_bn=True)

    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader_r_sub, valid_loader_f, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file, weights_only=False))

    # Test
    testing(conf.model_name, 'REAL', trainer, model, test_loader_r, df_test_r)
    testing(conf.model_name, 'FAKE', trainer, model, test_loader_f, df_test_f)

@dataclass
class TrainConf:
    model_name: str
    lr: float
    batch_size: int
    img_size: tuple[int, int]
    num_epochs: int

def ResNetConf():
    return TrainConf("ResNet50", 1e-2, 32, (32, 32), 10)

def MobileNetConf():
    return TrainConf("MobileNetV2", 1e-2, 32, (32, 32), 20)

def ViT16Conf():
    return TrainConf("ViT16", 1e-5, 64, (224, 224), 10)

if __name__== "__main__":

    starttime = time.time()

    model_name = "MobileNetV2"
    num_class = len(config.label_to_class)
    conf = MobileNetConf()
    finetune_layerwise(MobileNetV2, conf, real_raito=0.1)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))