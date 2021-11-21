import tensorflow as tf
from tensorflow import keras

from functools import partial
import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]

def process_image_file(filepath, size, top_percent=0.08, crop=True):
    img = cv2.imread(filepath)
    img = crop_top(img, percent=top_percent)
    if crop:
        img = central_crop(img)
    img = cv2.resize(img, (size, size))
    return img

def process_image_file_medusa(filepath, size):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = img.astype('float64')
    img -= img.mean()
    img /= img.std()
    return np.expand_dims(img, -1)

def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 480 or size[1] > 480:
        print(img.shape, size, ratio)

    img = cv2.resize(img, size)
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT,
                             (0, 0, 0))

    if img.shape[0] != 480 or img.shape[1] != 480:
        raise ValueError(img.shape, size)
    return img

_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)

def apply_augmentation(img):
    img = random_ratio_resize(img)
    img = _augmentation_transform.random_transform(img)
    return img

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files


class BalanceCovidDataset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            csv_file,
            is_training=True,
            batch_size=8,
            medusa_input_shape=(256, 256),
            input_shape=(480, 480),
            n_classes=2,
            num_channels=3,
            mapping={
                'negative': 0,
                'positive': 1,
            },
            shuffle=True,
            augmentation=apply_augmentation,
            covid_percent=0.5,
            class_weights=[1., 1.],
            top_percent=0.08,
            is_severity_model=False,
            is_medusa_backbone=False,
    ):
        'Initialization'
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.medusa_input_shape = medusa_input_shape
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = shuffle
        self.covid_percent = covid_percent
        self.class_weights = class_weights
        self.n = 0
        self.augmentation = augmentation
        self.top_percent = top_percent
        self.is_severity_model = is_severity_model
        self.is_medusa_backbone = is_medusa_backbone

        # If using MEDUSA backbone load images without crop
        if self.is_medusa_backbone:
            self.load_image = partial(process_image_file, top_percent=0, crop=False)
        else:
            self.load_image = process_image_file

        datasets = {}
        for key in self.mapping.keys():
            datasets[key] = []

        for l in self.dataset:
            datasets[l.split()[2]].append(l)
        
        if self.is_severity_model:
            self.datasets = [
                datasets['level2'], datasets['level1']
            ]
        elif self.n_classes == 2:
            self.datasets = [
                datasets['negative'], datasets['positive']
            ]
        elif self.n_classes == 3:
            self.datasets = [
                datasets['normal'] + datasets['pneumonia'],
                datasets['COVID-19'],
            ]
        else:
            raise Exception('Only binary or 3 class classification currently supported.')
        print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        model_inputs = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return model_inputs

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x = np.zeros((self.batch_size, *self.input_shape, self.num_channels))
        batch_y = np.zeros(self.batch_size)

        if self.is_medusa_backbone:
            batch_sem_x = np.zeros((self.batch_size, *self.medusa_input_shape, 1))

        batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) * self.batch_size]

        # upsample covid cases
        covid_size = max(int(len(batch_files) * self.covid_percent), 1)
        covid_inds = np.random.choice(np.arange(len(batch_files)),
                                      size=covid_size,
                                      replace=False)
        covid_files = np.random.choice(self.datasets[1],
                                       size=covid_size,
                                       replace=False)
        for i in range(covid_size):
            batch_files[covid_inds[i]] = covid_files[i]

        for i in range(len(batch_files)):
            sample = batch_files[i].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            image_file = os.path.join(self.datadir, folder, sample[1])
            x = self.load_image(
                image_file,
                self.input_shape[0],
                top_percent=self.top_percent,
            )

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation(x)

            x = x.astype('float32') / 255.0

            if self.is_medusa_backbone:
                sem_x = process_image_file_medusa(image_file, self.medusa_input_shape[0])
                batch_sem_x[i] = sem_x
            
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        class_weights = self.class_weights
        weights = np.take(class_weights, batch_y.astype('int64'))
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.n_classes)

        if self.is_medusa_backbone:
            return batch_sem_x, batch_x, batch_y, weights, self.is_training
        else:
            return batch_x, batch_y, weights, self.is_training
        