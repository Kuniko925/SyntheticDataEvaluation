from pathlib import Path

PROJECT_ROOT = Path(__file__).parent # Path to folder opened files

CFG = {
    "REAL": {"DB": "cifake1", "FileName": "real_real.csv", "ImageDir": "cifake1/train/REAL"},
    "FAKE1": {"DB": "cifake1", "FileName": "fake_real.csv", "ImageDir": "cifake1/train/FAKE"},
    "FAKE2": {"DB": "cifake2", "FileName": "real_real.csv", "ImageDir": "cifake2/train/FAKE"},
}

BEST_MODEL_PATH = {
    'FAKE1': {'MobileNetV2': 49, 'ResNet50': 43, 'ViT16': 46},
    'FAKE2': {'MobileNetV2': 42, 'ResNet50': 43, 'ViT16': 45},
    'REAL': {'MobileNetV2': 46, 'ResNet50': 40, 'ViT16': 47}
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
