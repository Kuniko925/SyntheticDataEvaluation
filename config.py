from pathlib import Path

PROJECT_ROOT = Path(__file__).parent # Path to folder opened files

CFG = {
    "REAL": {"DB": "cifake1", "FileName": "real_real.csv", "ImageDir": "cifake1/train/REAL"},
    "FAKE1": {"DB": "cifake1", "FileName": "fake_real.csv", "ImageDir": "cifake1/train/FAKE"},
    "FAKE2": {"DB": "cifake2", "FileName": "real_real.csv", "ImageDir": "cifake2/train/FAKE"},
}

MODELS = ["ResNet50", "MobileNetV2", "ViT16"]

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

def make_class_to_label() -> dict[str, int]:
    return {class_name: label for label, class_name in label_to_class.items()}

def build_res_filepath(data_type, model, test_setting='REAL'):
    return PROJECT_ROOT / f"results/{model}_{data_type}_{test_setting}.csv"