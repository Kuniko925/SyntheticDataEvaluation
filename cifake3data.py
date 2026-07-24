import data
import utils
from train_utils import set_seed
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import torch
import config
import time
import train_settings

def run_training(model_class, conf, db):

    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset(db)
    train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f = data.get_dataloaders(
        df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    def preparation(model_name, db):
        # Preparation
        model_save_directory = config.PROJECT_ROOT / db / f"cifake3/{model_name}/{db}/" # e.g., cifake1/MobileNet/FAKE/
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

if __name__== "__main__":

    set_seed(42)
    starttime = time.time()

    db = 'MIX'
    run_training(MobileNetV2, train_settings.MobileNetConf(), db)
    run_training(ResNet50Model, train_settings.ResNetConf(), db)
    run_training(ViT16, train_settings.ViT16Conf(), db)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))