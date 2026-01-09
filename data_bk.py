import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image

label_to_class={0: 'airplane',
             1: 'automobile',
             2: 'bird',
             3: 'cat',
             4: 'deer',
             5: 'dog',
             6: 'frog',
             7: 'horse',
             8: 'ship',
             9: 'truck'}

class_to_label = {'airplane': 0,
               'automobile': 1,
               'bird': 2,
               'cat': 3,
               'deer': 4,
               'dog': 5,
               'frog': 6,
               'horse': 7,
               'ship': 8,
               'truck': 9}

def get_id_to_class():
    return label_to_class

class CIFAKEDataSet():
    def __init__(self, root):
        self.root = root
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
        df["label"] = df["class name"].map(class_to_label)
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

def get_dataloader(df, img_size, batch_size, label_encoder, train=False, brightness=0.2, contrast=0.2): # Train = True, Valid or Test = False
    transform = None
    df["label"] = label_encoder.transform(df["label"])
    
    if train:
        transform = transforms.Compose([
                    transforms.Resize(img_size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=0.2, hue=0.2),
                    transforms.RandomRotation(20),
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25),
                ])
    else:
        transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader


class AdTransDataset(Dataset):
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
        distance = self.dataframe.iloc[idx]["distance from real"]
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), real, distance


def get_ad_dataloader(df, img_size, batch_size, label_encoder, train=False): # Train = True, Valid or Test = False
    transform = None
    df["label"] = label_encoder.transform(df["label"])
    
    if train:
        transform = transforms.Compose([
                    transforms.Resize(img_size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomRotation(20),
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25),
                ])
    else:
        transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    
    dataset = AdTransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader
