import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

class BalanceDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset,
                 is_training=True,
                 batch_size=8,
                 input_shape=(224,224),
                 n_classes=3,
                 num_channels=3,
                 mapping={'normal': 0, 'pneumonia': 1, 'COVID-19': 2},
                 shuffle=True,
                 augmentation=True,
                 datadir='data',
                 class_weights=[1., 1., 25.]
                 ):
        'Initialization'
        self.datadir = datadir
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.n = 0
        self.class_weights = class_weights

        if augmentation:
            self.augmentation = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=(0.9, 1.1),
                fill_mode='constant',
                cval=0.,
            )

        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in dataset:
            datasets[l.split()[-1]].append(l)
        self.datasets = [
            datasets['normal'] + datasets['pneumonia'],
            datasets['COVID-19'],
        ]
        print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y, weights = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return batch_x, batch_y, weights

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx*self.batch_size : (idx+1)*self.batch_size]
        batch_files[np.random.randint(self.batch_size)] = np.random.choice(self.datasets[1])

        for i in range(self.batch_size):
            sample = batch_files[i].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join(self.datadir, folder, sample[1]))
            x = cv2.resize(x, self.input_shape)

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation.random_transform(x)

            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        weights = np.take(self.class_weights, batch_y.astype('int64'))

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes), weights


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset,
                 is_training=True,
                 batch_size=8,
                 input_shape=(224,224),
                 n_classes=3,
                 num_channels=3,
                 mapping={'normal': 0, 'pneumonia': 1, 'COVID-19': 2},
                 shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset, random_state=0)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join('data', folder, sample[1]))
            x = cv2.resize(x, self.input_shape)
            x = x.astype('float32') / 255.0
            #y = int(sample[1])
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
