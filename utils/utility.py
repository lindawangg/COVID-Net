from model.resnet1 import ResNet50_1
from model.resnet2 import ResNet50_2

def select_resnet_type(name,classes,model_semantic):
    if(name == "resnet1"):
        return ResNet50_1(classes=classes, model_semantic=model_semantic)
    elif(name == "resnet2"):
        return ResNet50_2(classes=classes, model_semantic=model_semantic)