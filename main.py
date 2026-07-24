import data
from train_utils import *
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time
from train_settings import *


def run_training(model_class, conf, db, seed,):

    num_class = len(config.label_to_class)
    set_seed(seed)

    # Data
    df_train, df_valid = data.get_train_data(db)
    df_test_real = data.get_test_data('REAL')
    df_test_fake1 = data.get_test_data('FAKE1')
    df_test_fake2 = data.get_test_data('FAKE2')

    train_loader = data.get_train_loader(df_train, conf.batch_size, conf.img_size)
    valid_loader = data.get_test_loader(df_valid, conf.batch_size, conf.img_size)

    test_loader_real = data.get_test_loader(df_test_real, conf.batch_size, conf.img_size)
    test_loader_fake1 = data.get_test_loader(df_test_fake1, conf.batch_size, conf.img_size)
    test_loader_fake2 = data.get_test_loader(df_test_fake2, conf.batch_size, conf.img_size)

    # Training Preparation, db name refers to training dataset
    save_dir = (config.PROJECT_ROOT / db / f"{conf.model_name}_{seed}")
    model_save_directory = prepare_save_directory(save_dir)

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load the best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file))

    # Test
    output_path = config.PROJECT_ROOT / f'results/{conf.model_name}_{db}_REAL_{seed}.csv'
    evaluate_and_save(model, trainer, test_loader_real, df_test_real, output_path)

    output_path = config.PROJECT_ROOT / f'results/{conf.model_name}_{db}_FAKE1_{seed}.csv'
    evaluate_and_save(model, trainer, test_loader_fake1, df_test_fake1, output_path)

    output_path = config.PROJECT_ROOT / f'results/{conf.model_name}_{db}_FAKE2_{seed}.csv'
    evaluate_and_save(model, trainer, test_loader_fake2, df_test_fake2, output_path)

if __name__== "__main__":

    seeds = [12, 123, 1234]
    DB = ['REAL', 'FAKE1', 'FAKE2']
    EXPERIMENT_MODELS = [
        (MobileNetV2, MobileNetConf()),
        (ResNet50Model, ResNetConf()),
        (ViT16, ViT16Conf()),
    ]

    starttime = time.time()

    for seed in seeds:
        for db in DB:
            for model_class, conf in EXPERIMENT_MODELS:
                run_training(model_class=model_class, conf=conf, db=db, seed=seed,)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
