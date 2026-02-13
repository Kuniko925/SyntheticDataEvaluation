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

class CIFAKEDataSet():
    def __init__(self, root):
        self.root = root
        self.class_to_label = config.make_class_to_label()
    def create_dataset(self, d_type, save_filepath=None):
        rf_dir = os.listdir(f"{self.root}{d_type}/") # Real or Fake
        allfiles = []
        for rf in rf_dir:
            class_dir = os.listdir(f"{self.root}{d_type}/{rf}/")
            for dir in class_dir:
                files = os.listdir(f"{self.root}{d_type}/{rf}/{dir}/")
                for f in files:
                    filepath = f"{self.root}{d_type}/{rf}/{dir}/{f}"
                    allfiles.append({"filepath":filepath, "class name": dir, "rf":rf})
                    
        df = pd.DataFrame(allfiles)
        df.loc[df["rf"] == "REAL", "real"] = 1
        df.loc[df["rf"] == "FAKE", "real"] = 0
        df["real"] = df["real"].astype(int)
        df["label"] = df["class name"].map(self.class_to_label)
        if save_filepath is not None:
            df.to_csv(save_filepath, index=False)
        return df

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
