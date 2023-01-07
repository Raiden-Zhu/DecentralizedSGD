

from .resnet import ResNet18
from .alexnet import alexnet
from .densenet import DenseNet

def load_model(name, outputsize, pretrained=None):

    if pretrained:
        pretrained = True
    else:
        pretrained = False

    if name.lower() == 'resnet18':
        model = ResNet18(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == 'alexnet':
        model = alexnet(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == "densenet":
        model = DenseNet(num_classes=outputsize)

    return model



