from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten

def build_COVIDNet(num_classes=3, flatten=True, checkpoint=''):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    if flatten:
        x = Flatten()(x)
    else:
        x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if len(checkpoint):
        model.load_weights(checkpoint)
    return model
