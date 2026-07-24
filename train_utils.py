import os
import random
import numpy as np
import torch
from pathlib import Path
import utils
import pandas as pd

# to reproduce
def set_seed(seed)-> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

# To get samples based on the fract ratio keeping label groups
def stratified_sample(df, label_col="label", frac=0.10, random_state=42) -> pd.DataFrame:
    return df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)

# To create a directory for saving the best model
def prepare_save_directory(save_directory: Path) -> Path:
    utils.create_directory(save_directory)
    utils.delete_subfolders(save_directory)
    return save_directory

# To save test results
def evaluate_and_save(model, trainer, test_loader, test_df, output_path,):
    result_df = test_df.copy()
    result_df["preds"] = trainer.evaluate(model, test_loader)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df