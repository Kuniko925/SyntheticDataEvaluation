import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import config

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
    
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader