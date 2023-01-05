

from .resnet import ResNet18
from .alexnet import AlexNet
from .densenet import DenseNet

def load_model(name, inputsize, outputsize):

    if name.lower() == 'resnet18':
        model = ResNet18(num_classes=outputsize)
    if name.lower() == 'alexnet':
        model = AlexNet(num_classes=outputsize)
    if name.lower() == "densenet":
        model = DenseNet(num_classes=outputsize)

    return model



