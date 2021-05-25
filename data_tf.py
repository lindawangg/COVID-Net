import os
import numpy as np
import tensorflow as tf

import augmentations


_CLASS_MAPS = {
    2: {'negative': 0, 'positive': 1},
    3: {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
}


class COVIDxDataset:
    """COVIDx dataset class, which handles construction of train/test datasets"""
    def __init__(self, data_dir, num_classes, image_size=480, sem_image_size=256,
                 max_translation=20, max_rotation=10, max_shear=0.15, max_pixel_shift=10,
                 max_pixel_scale_change=0.1, class_weights=None, shuffle_buffer=10000):
        # General parameters
        self.data_dir = data_dir
        self.image_size = image_size
        self.sem_image_size = sem_image_size
        self.shuffle_buffer = shuffle_buffer
        self.num_classes = num_classes
        self.class_map = _CLASS_MAPS[num_classes]
        self.class_weights = tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None

        # Augmentation parameters
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_pixel_shift = max_pixel_shift
        self.max_pixel_scale_change = max_pixel_scale_change

    def train_dataset(self, train_split_file, batch_size=1):
        """Returns training dataset"""
        return self._make_dataset(train_split_file, batch_size, True)

    def test_dataset(self, test_split_file, batch_size=1):
        """Returns test dataset"""
        return self._make_dataset(test_split_file, batch_size, False)

    def _make_dataset(self, split_file, batch_size, is_training, balanced=True):
        """Creates COVIDX-CT dataset for train or val split"""
        files, classes = self._get_files(split_file, is_training)
        count = len(files)

        if is_training:
            # Create balanced dataset if required
            if balanced:
                files = np.asarray(files)
                classes = np.asarray(classes, dtype=np.int32)
                class_nums = np.unique(classes)
                class_wise_datasets = []
                for cls in class_nums:
                    indices = np.where(classes == cls)[0]
                    class_wise_datasets.append(
                        tf.data.Dataset.from_tensor_slices((files[indices], classes[indices])))
                class_weights = [1.0 / len(class_nums) for _ in class_nums]
                dataset = tf.data.experimental.sample_from_datasets(
                    class_wise_datasets, class_weights)

            # Shuffle and repeat
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
            dataset = dataset.repeat()
        else:
            dataset = tf.data.Dataset.from_tensor_slices((files, classes))

        # Create and apply map function
        load_and_process = self._get_load_and_process_fn(is_training)
        dataset = dataset.map(load_and_process)

        # Batch and prefetch data
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, count, batch_size

    def _get_load_and_process_fn(self, is_training):
        """Creates map function for TF dataset"""
        def load_and_process(path, label):
            # Load image and ensure grayscale
            base_image = tf.image.decode_image(tf.io.read_file(path), channels=0)
            base_image = tf.reduce_mean(base_image, axis=-1, keepdims=True)
            base_image.set_shape([None, None, 1])

            # Apply augmentations
            if is_training:
                base_image = self._augment_image(base_image)
            base_image = tf.cast(base_image, tf.float32)

            # Resize, stack to 3-channel, and scale to [0, 1]
            image = tf.image.resize(base_image, [self.image_size, self.image_size])
            image = tf.image.grayscale_to_rgb(image)
            image = image / 255.0

            # Resize and Z-score normalize semantic image
            sem_image = tf.image.resize(base_image, [self.sem_image_size, self.sem_image_size])
            sem_image = (sem_image - tf.reduce_mean(sem_image))/tf.math.reduce_std(sem_image)
            image = (image - tf.reduce_mean(image))/tf.math.reduce_std(image)

            # Convert label to one-hot encoded vector
            oh_label = tf.one_hot(label, depth=self.num_classes)

            # Get class weight
            if self.class_weights is not None:
                weight = tf.gather(self.class_weights, label)
                return {'image': image, 'sem_image': sem_image, 'label': oh_label, 'weight': weight}

            return {'image': image, 'sem_image': sem_image, 'label': oh_label}

        return load_and_process

    def _augment_image(self, image):
        """Apply augmentations to image"""
        image = augmentations.random_shear(image, self.max_shear)
        image = augmentations.random_rotation(image, self.max_rotation)
        image = augmentations.random_translation(image, self.max_translation)
        image = augmentations.random_shift_and_scale(image, self.max_pixel_shift, self.max_pixel_scale_change)
        image = tf.image.random_flip_left_right(image)
        image= tf.image.random_brightness(image, 0.1, seed=None)
        return image

    def _get_files(self, split_file, is_training):
        """Gets image filenames and classes"""
        files, classes = [], []
        img_dir = os.path.join(self.data_dir, 'train' if is_training else 'test')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                split = line.split()
                if split[-1] == 'sirm':
                    files.append(os.path.join(img_dir, split[2]))
                    classes.append(self.class_map[split[3]])
                else:
                    files.append(os.path.join(img_dir, split[1]))
                    classes.append(self.class_map[split[2]])
        return files, classes
