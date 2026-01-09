from pathlib import Path

PROJECT_ROOT = Path(__file__).parent # Path to folder opened files

CFG = {
    "REAL": {"DB": "cifake1", "FileName": "real_real.csv", "ImageDir": "cifake1/train/REAL"},
    "FAKE1": {"DB": "cifake1", "FileName": "fake_real.csv", "ImageDir": "cifake1/train/FAKE"},
    "FAKE2": {"DB": "cifake2", "FileName": "real_real.csv", "ImageDir": "cifake2/train/FAKE"},
}

MODELS = ["ResNet50Model", "MobileNetV2", "ViT16"]

label_to_class = {0: 'airplane',
                      1: 'automobile',
                      2: 'bird',
                      3: 'cat',
                      4: 'deer',
                      5: 'dog',
                      6: 'frog',
                      7: 'horse',
                      8: 'ship',
                      9: 'truck'}

def build_res_filepath(dataset, model=None):
    if model is None:
        return [
            f"{CFG[dataset]['DB']}/results/{model}_{CFG[dataset]['FileName']}" for model in MODELS
        ]
    else:
        return f"{CFG[dataset]['DB']}/results/{model}_{CFG[dataset]['FileName']}"
