

from .resnet import ResNet18
from .alexnet import alexnet
from .densenet import DenseNet121 
from .resnet_modify import resnet18 as resnet18_m
from .resnet_modify import resnet34 as resnet34_m
from .densenet_modify import densenet121 as densenet121_m

def load_model(name, outputsize, pretrained=None):

    if pretrained:
        pretrained = True
    else:
        pretrained = False

    if name.lower() == 'resnet18':
        model = ResNet18(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == 'resnet18_m':
        model = resnet18_m(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == 'resnet34_m':
        model = resnet34_m(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == 'alexnet':
        model = alexnet(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == "densenet121":
        model = DenseNet121(num_classes=outputsize, pretrained=pretrained)
    if name.lower() == "densenet121_m":
        model = densenet121_m(num_classes=outputsize, pretrained=False)

    return model



