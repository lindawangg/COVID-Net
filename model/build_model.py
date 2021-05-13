from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D


def build_UNet2D_4L(inp_shape, k_size=3, trainable=True):
    merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution2D(
        filters=32, kernel_size=k_size, padding='same', activation='relu', name="sem/1", trainable=trainable)(data)
    conv1 = Convolution2D(
        filters=32, kernel_size=k_size, padding='same', activation='relu', name="sem/2", trainable=trainable)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="sem/3", trainable=trainable)(conv1)

    conv2 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/4", trainable=trainable)(pool1)
    conv2 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/5", trainable=trainable)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="sem/6", trainable=trainable)(conv2)

    conv3 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/7", trainable=trainable)(pool2)
    conv3 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/8", trainable=trainable)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="sem/9", trainable=trainable)(conv3)

    conv4 = Convolution2D(
        filters=128, kernel_size=k_size, padding='same', activation='relu', name="sem/10", trainable=trainable)(pool3)
    conv4 = Convolution2D(
        filters=128, kernel_size=k_size, padding='same', activation='relu', name="sem/11", trainable=trainable)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="sem/12", trainable=trainable)(conv4)

    conv5 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/13", trainable=trainable)(pool4)

    up1 = UpSampling2D(size=(2, 2), name="sem/14", trainable=trainable)(conv5)
    conv6 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/15", trainable=trainable)(up1)
    conv6 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/16", trainable=trainable)(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis, name="sem/17")
    conv6 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/18", trainable=trainable)(merged1)

    up2 = UpSampling2D(size=(2, 2), name="sem/19", trainable=trainable)(conv6)
    conv7 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/20", trainable=trainable)(up2)
    conv7 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/21", trainable=trainable)(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis, name="sem/22")
    conv7 = Convolution2D(
        filters=256, kernel_size=k_size, padding='same', activation='relu', name="sem/23", trainable=trainable)(merged2)

    up3 = UpSampling2D(size=(2, 2), name="sem/24", trainable=trainable)(conv7)
    conv8 = Convolution2D(
        filters=128, kernel_size=k_size, padding='same', activation='relu', name="sem/25", trainable=trainable)(up3)
    conv8 = Convolution2D(
        filters=128, kernel_size=k_size, padding='same', activation='relu', name="sem/26", trainable=trainable)(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis, name="sem/27")
    conv8 = Convolution2D(
        filters=128, kernel_size=k_size, padding='same', activation='relu', name="sem/28", trainable=trainable)(merged3)

    up4 = UpSampling2D(size=(2, 2), name="sem/29", trainable=trainable)(conv8)
    conv9 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/30", trainable=trainable)(up4)
    conv9 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/31", trainable=trainable)(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis, name="sem/32")
    conv9 = Convolution2D(
        filters=64, kernel_size=k_size, padding='same', activation='relu', name="sem/33", trainable=trainable)(merged4)

    conv10 = Convolution2D(
        filters=1, kernel_size=k_size, padding='same', activation='sigmoid', name="sem/34", trainable=trainable)(conv9)

    output = conv10
    model = Model(data, output)
    return model
