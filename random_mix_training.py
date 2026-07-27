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
    df_train_real, df_valid_real = data.get_train_data('REAL')
    df_train_fake1, df_valid_fake1 = data.get_train_data('FAKE1')

    # train
    df_train_real_sub = stratified_sample(df_train_real, label_col="label", frac=real_ratio, random_state=seed)
    df_train_fake1_sub = stratified_sample(df_train_fake1, label_col="label", frac=(1-real_ratio), random_state=seed)

    # valid
    df_valid_real_sub = stratified_sample(df_valid_real, label_col="label", frac=real_ratio, random_state=seed)
    df_valid_fake1_sub = stratified_sample(df_valid_fake1, label_col="label", frac=(1-real_ratio), random_state=seed)

    # test
    df_test = data.get_test_data('REAL')

    # concat
    df_train = pd.concat([df_train_real_sub, df_train_fake1_sub], axis=0, ignore_index=True,)
    df_valid = pd.concat([df_valid_real_sub, df_valid_fake1_sub], axis=0, ignore_index=True,)

    train_loader = data.get_train_loader(df_train, conf.batch_size, conf.img_size)
    valid_loader = data.get_test_loader(df_valid, conf.batch_size, conf.img_size)
    test_loader = data.get_test_loader(df_test, conf.batch_size, conf.img_size)

    # Training Preparation
    str_real_ratio = str(int(real_ratio * 100))
    model_save_directory = config.PROJECT_ROOT / f"MIX/{conf.model_name}_{str_real_ratio}_{seed}"

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load the best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file, weights_only=False))

    # Test
    output_path = config.PROJECT_ROOT / f"results/{conf.model_name}_MIX_{int(real_ratio * 100)}_{seed}.csv"
    evaluate_and_save(model, trainer, test_loader, df_test, output_path)

if __name__== "__main__":

    starttime = time.time()

    seeds = [12, 123, 1234]
    real_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,]
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