from torchvision.models import resnet18
from torch import nn

def DogVsCatWithResNet18():
    model = resnet18(weights="IMAGENET1K_V1")

    for params in model.parameters():
        params.requires_grad = False

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2) # There are 2 things to be classified (dog and cat)
    return model