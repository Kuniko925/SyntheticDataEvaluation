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
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset(db)
    train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = data.get_dataloaders(
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    def testing(model_name, db, test_setting, trainer, model, test_loader, df_test):
        preds = trainer.evaluate(model, test_loader)
        df_test['preds'] = preds
        df_test.to_csv(config.PROJECT_ROOT / f'results/{model_name}_{db}_{test_setting}_{seed}.csv', index=False) # e.g., results/MobileNet_FAKE_REAL.csv/

    # Training Preparation
    save_dir = (config.PROJECT_ROOT / db/ f"{conf.model_name}_{db}_{seed}")
    model_save_directory = prepare_save_directory(save_dir)

    if db == "REAL":
        train_loader = train_loader_r
        valid_loader = valid_loader_r
    elif db == "FAKE1" or db == "FAKE2":
        train_loader = train_loader_f
        valid_loader = valid_loader_f

    # Training
    model = model_class(num_class)
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(model, train_loader, valid_loader, model_save_directory, epochs=conf.num_epochs, lr=conf.lr)

    # Load best model
    model = model_class(num_class)
    model.load_state_dict(torch.load(best_val_file))

    # Test
    testing(conf.model_name, db, 'REAL', trainer, model, test_loader_r, df_test_r)
    testing(conf.model_name, db, 'FAKE', trainer, model, test_loader_f, df_test_f)

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
