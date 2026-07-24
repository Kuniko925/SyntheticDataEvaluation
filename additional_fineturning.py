import data
from finetune import CNNModelTrainer, TransModelTrainer
from model import ResNet50Model, MobileNetV2, ViT16
import config
import time
from train_utils import *
from train_settings import *


def additional_finetune(model_class, conf, saving_epochs = [10], seed=42):

    num_class = len(config.label_to_class)

    # Data
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = data.get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=0.10, random_state=42)
    df_valid_r_sub = stratified_sample(df_valid_r, label_col="label", frac=0.10, random_state=42)

    train_loader_r_sub, valid_loader_r_sub, test_loader_r, _, _, _ = data.get_dataloaders(
        df_train_r_sub, df_valid_r_sub, df_test_r, df_train_f, df_valid_f, df_test_f, conf.batch_size, conf.img_size)

    # Training Preparation
    model_save_directory = prepare_save_directory(config.PROJECT_ROOT / f"Unfreeze/{conf.model_name}")

    # Training
    model = model_class(num_class)
    best_val_file = config.PROJECT_ROOT / f'FAKE1/{conf.model_name}_FAKE1_seed_{seed}_best.pt'
    model.load_state_dict(torch.load(best_val_file))
    trainer = TransModelTrainer() if conf.model_name == 'ViT16' else CNNModelTrainer()
    best_val_file = trainer.fit(
        model, train_loader_r_sub, valid_loader_r_sub, model_save_directory, epochs=150, lr=conf.lr, saving_epochs=saving_epochs)


    for saving_epoch in saving_epochs:
        # Load best model
        best_val_file = config.PROJECT_ROOT / f'Unfreeze/{conf.model_name}/model_{str(saving_epoch-1)}.pt'
        model = model_class(num_class)
        model.load_state_dict(torch.load(best_val_file, weights_only=False))

        # Test
        df_test_r['preds'] = trainer.evaluate(model, test_loader_r)
        df_test_r.to_csv(config.PROJECT_ROOT / f'results/{conf.model_name}_Unfreeze_{str(str(saving_epoch))}.csv', index=False)

if __name__== "__main__":

    starttime = time.time()
    seeds = [12, 123, 1234]
    EXPERIMENT_MODELS = [
        (MobileNetV2, MobileNetConf()),
        (ResNet50Model, ResNetConf()),
        (ViT16, ViT16Conf()),
    ]
    saving_epochs = list(range(10, 151, 10))

    for seed in seeds:
        for model_class, conf in EXPERIMENT_MODELS:
            additional_finetune(model_class=model_class, conf=conf, saving_epochs=saving_epochs, seed=seed,)

    endtime = time.time()
    interval = endtime - starttime
    print("running time = %dh %dm %ds" % (int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))