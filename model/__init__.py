from .resnet0 import ResNet50_0
from .resnet1 import ResNet50_1
from .resnet2 import ResNet50_2
from .build_model import build_UNet2D_4L


def build_resnet_attn_model(name, classes, model_semantic):
    if name[:7] == 'resnet0':
        mode_char = name.split('_')[1].lower()
        if mode_char == 'r':
            sem_mode = 'residual'
        elif mode_char == 'm':
            sem_mode = 'mask'
        else:
            raise ValueError('Invalid conection mode for ResNet0 model: {}'.format(mode_char))
        return ResNet50_0(classes=classes, model_semantic=model_semantic, semantic_mode=sem_mode)
    elif name == 'resnet1':
        return ResNet50_1(classes=classes, model_semantic=model_semantic)
    elif name == 'resnet2':
        return ResNet50_2(classes=classes, model_semantic=model_semantic)
    else:
        raise ValueError('Unrecognized ResNet type: {}'.format(name))
