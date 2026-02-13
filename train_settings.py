from dataclasses import dataclass

@dataclass
class TrainConf:
    model_name: str
    lr: float
    batch_size: int
    img_size: tuple[int, int]
    num_epochs: int

def ResNetConf():
    return TrainConf("ResNet50", 1e-2, 32, (32, 32), 50)

def MobileNetConf():
    return TrainConf("MobileNetV2", 1e-2, 32, (32, 32), 50)

def ViT16Conf():
    return TrainConf("ViT16", 1e-5, 64, (224, 224), 50)