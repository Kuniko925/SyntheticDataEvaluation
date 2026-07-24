import data
from train_utils import *
from train_settings import *
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time
import pandas as pd

def mix_finetune(model_class, conf, real_ratio=0.1, seed=42,):

    set_seed(seed)
    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=real_ratio, random_state=seed)
    df_train_f_sub = stratified_sample(df_train_f, label_col="label", frac=(1-real_ratio), random_state=seed)
    df_valid_r_sub = stratified_sample(df_valid_r, label_col="label", frac=real_ratio, random_state=seed)
    df_valid_f_sub = stratified_sample(df_valid_f, label_col="label", frac=(1-real_ratio), random_state=seed)
    df_train = pd.concat([df_train_r_sub, df_train_f_sub], axis=0)
    df_valid = pd.concat([df_valid_r_sub, df_valid_f_sub], axis=0)

    train_loader, valid_loader, test_loader_r, _, _, _ = data.get_dataloaders(
        df_train, df_valid, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    # Training Preparation
    model_save_directory = prepare_save_directory(config.PROJECT_ROOT / "MIX" / conf.model_name / str(int(real_ratio * 100)))

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file, weights_only=False))

    # Test
    df_test_r['preds'] = trainer.evaluate(model, test_loader_r)
    df_test_r.to_csv(config.PROJECT_ROOT / f"results/{conf.model_name}_mix_{int(real_ratio * 100)}_seed_{seed}.csv", index=False)

if __name__== "__main__":

    starttime = time.time()

    seeds = [12, 123, 1234]
    real_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    EXPERIMENT_MODELS = [
        (MobileNetV2, MobileNetConf()),
        (ResNet50Model, ResNetConf()),
        (ViT16, ViT16Conf()),
    ]

    for seed in seeds:
        for model_class, conf in EXPERIMENT_MODELS:
            for real_ratio in real_ratios:
                mix_finetune(model_class=model_class, conf=conf, real_ratio=real_ratio, seed=seed)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))