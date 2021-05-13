from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D


def build_UNet2D_4L(inp_shape, k_size=3):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu',name="sem/1")(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu',name="sem/2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name="sem/3")(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/4")(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/5")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name="sem/6")(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/7")(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/8")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name="sem/9")(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu',name="sem/10")(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu',name="sem/11")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name="sem/12")(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/13")(pool4)

    up1 = UpSampling2D(size=(2, 2),name="sem/14")(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/15")(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/16")(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis,name="sem/17")
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/18")(merged1)

    up2 = UpSampling2D(size=(2, 2),name="sem/19")(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/20")(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/21")(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis,name="sem/22")
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu',name="sem/23")(merged2)

    up3 = UpSampling2D(size=(2, 2),name="sem/24")(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu',name="sem/25")(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu',name="sem/26")(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis,name="sem/27")
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu',name="sem/28")(merged3)

    up4 = UpSampling2D(size=(2, 2),name="sem/29")(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/30")(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/31")(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis,name="sem/32")
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu',name="sem/33")(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid',name="sem/34")(conv9)

    output = conv10
    model = Model(data, output)
    return model
