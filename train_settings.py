from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConf:
    model_name: str
    lr: float
    batch_size: int
    img_size: tuple[int, int]
    num_epochs: int

def ResNetConf() -> TrainConf:
    return TrainConf(
        model_name="ResNet50",
        lr=1e-2,
        batch_size=32,
        img_size=(32, 32),
        num_epochs=30,
    )


def MobileNetConf() -> TrainConf:
    return TrainConf(
        model_name="MobileNetV2",
        lr=1e-2,
        batch_size=32,
        img_size=(32, 32),
        num_epochs=30,
    )

def ViT16Conf() -> TrainConf:
    return TrainConf(
        model_name="ViT16",
        lr=1e-5,
        batch_size=64,
        img_size=(224, 224),
        num_epochs=30,
    )