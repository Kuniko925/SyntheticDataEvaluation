import data
from train_utils import *
from train_settings import *
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time


def run_training(model_class, conf, target_class=6, seed=42):

    set_seed(seed)
    num_class = len(config.label_to_class)

    # Data
    df_train_real, df_valid_real = data.get_train_data('REAL')
    df_train_fake1, df_valid_fake1 = data.get_train_data('FAKE1')
    df_test = data.get_test_data('REAL')

    df_train_real_target = df_train_real[df_train_real['label'] == target_class]
    target_index = df_train_fake1[df_train_fake1['label'] == target_class].index
    df_train_fake1.drop(target_index, inplace=True)
    df_train = pd.concat([df_train_fake1, df_train_real_target], axis=0, ignore_index=True, )

    df_valid_real_target = df_valid_real[df_valid_real['label'] == target_class]
    target_index = df_valid_fake1[df_valid_fake1['label'] == target_class].index
    df_valid_fake1.drop(target_index, inplace=True)
    df_valid = pd.concat([df_valid_fake1, df_valid_real_target], axis=0, ignore_index=True,)

    train_loader = data.get_train_loader(df_train, conf.batch_size, conf.img_size)
    valid_loader = data.get_test_loader(df_valid, conf.batch_size, conf.img_size)
    test_loader = data.get_test_loader(df_test, conf.batch_size, conf.img_size)

    # Training Preparation
    model_save_directory = config.PROJECT_ROOT / f"results/{conf.model_name}_ONE_{seed}"

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file))

    # Test
    output_path = config.PROJECT_ROOT / f"results/{conf.model_name}_ONE_{seed}.csv"
    evaluate_and_save(model, trainer, test_loader, df_test, output_path)


if __name__== "__main__":

    starttime = time.time()

    seeds = [12, 123, 1234]
    EXPERIMENT_MODELS = [
        (MobileNetV2, MobileNetConf()),
        (ResNet50Model, ResNetConf()),
        (ViT16, ViT16Conf()),
    ]

    for seed in seeds:
        for model_class, conf in EXPERIMENT_MODELS:
            run_training(model_class=model_class, conf=conf, target_class=6, seed=seed,)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))