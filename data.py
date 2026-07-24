import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

seed = 42
class TransDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        filepath = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(filepath).convert("RGB")
        label = self.dataframe.iloc[idx]["label"]
        real = self.dataframe.iloc[idx]["real"]
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), real

def get_dataloader(df, img_size, batch_size, label_encoder, train=False): # Train = True, Valid or Test = False
    df["label"] = label_encoder.transform(df["label"])
    
    if train:
        transform = transforms.Compose([
                    transforms.Resize(img_size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                ])
    else:
        transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])

    drop_last = True if train else False
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True, drop_last=drop_last)
    return loader

def load_csv_and_fix_filepath(csv_path: str, project_root) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["filepath"] = df["filepath"].str.replace(
        r'^\.\./\.\./dataset/CIFAKE/',
        str(project_root) + "/",
        regex=True
    )
    return df

def add_image_column(df: pd.DataFrame, filepath_col: str = "filepath", image_col: str = "image") -> pd.DataFrame:
    df = df.copy()
    df[image_col] = df[filepath_col].apply(os.path.basename)
    return df

def get_train_loader(df, batch_size=32, img_size=(32, 32)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    return loader

def get_test_loader(df, batch_size=32, img_size=(32, 32)):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
    return loader

def get_train_data(db):
    test_size = 0.2
    save_filepath = config.PROJECT_ROOT / 'train.csv'
    df = pd.read_csv(save_filepath)
    df = df[df['rf'] == db]
    df_train, df_valid = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=df['label'],)
    return df_train, df_valid

def get_test_data(db):
    save_filepath = config.PROJECT_ROOT / 'test.csv'
    df = pd.read_csv(save_filepath)
    df = df[df['rf'] == db]
    return df

def get_dataset(db):
    test_size = 0.2

    if db == "MIX":
        save_filepath = config.PROJECT_ROOT / f'cifake3/train.csv'
    else:
        save_filepath = config.PROJECT_ROOT / f'{config.CFG[db]["DB"]}/train.csv'

    df = load_csv_and_fix_filepath(save_filepath, config.PROJECT_ROOT)
    df_real = df[df['rf'] == 'REAL'].copy()
    df_fake = df[df['rf'] == 'FAKE'].copy()
    df_train_r, df_valid_r = train_test_split(df_real, test_size=test_size, random_state=seed, shuffle=True)
    df_train_f, df_valid_f = train_test_split(df_fake, test_size=test_size, random_state=seed, shuffle=True)

    if db == "MIX":
        save_filepath = config.PROJECT_ROOT / f'cifake3/test.csv'
    else:
        save_filepath = config.PROJECT_ROOT / f'{config.CFG[db]["DB"]}/test.csv'

    df_test = load_csv_and_fix_filepath(save_filepath, config.PROJECT_ROOT)
    df_test_r = df_test[df_test['rf'] == 'REAL'].copy()
    df_test_f = df_test[df_test['rf'] == 'FAKE'].copy()

    df_train_r = add_image_column(df_train_r)
    df_valid_r = add_image_column(df_valid_r)
    df_train_f = add_image_column(df_train_f)
    df_valid_f = add_image_column(df_valid_f)
    df_test_r = add_image_column(df_test_r)
    df_test_f = add_image_column(df_test_f)

    return df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f



def get_dataloaders(df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f, batch_size=32, img_size=(32, 32)):

    label_encoder = LabelEncoder()
    label_encoder.fit(df_train_r["label"])
    train_loader_r = get_dataloader(df_train_r, img_size, batch_size, label_encoder, train=True)
    valid_loader_r = get_dataloader(df_valid_r, img_size, batch_size, label_encoder, train=False)
    test_loader_r = get_dataloader(df_test_r, img_size, batch_size, label_encoder, train=False)
    train_loader_f = get_dataloader(df_train_f, img_size, batch_size, label_encoder, train=True)
    valid_loader_f = get_dataloader(df_valid_f, img_size, batch_size, label_encoder, train=False)
    test_loader_f = get_dataloader(df_test_f, img_size, batch_size, label_encoder, train=False)

    return train_loader_r, valid_loader_r, test_loader_r, train_loader_f, valid_loader_f, test_loader_f

def stratified_sample(df, label_col="label", frac=0.20, random_state=seed):
    return df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)

def make_subset(real_ratio):
    df_train_r, df_valid_r, df_test_r, df_train_f, df_valid_f, df_test_f = get_dataset('FAKE1')
    df_train_r_sub = stratified_sample(df_train_r, label_col="label", frac=real_ratio)
    df_train_f_sub = stratified_sample(df_train_f, label_col="label", frac=(1-real_ratio))
    return pd.concat([df_train_r_sub, df_train_f_sub], ignore_index=True)