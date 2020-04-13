"""Perform transfer-learning for offset stratification with a provided COVID-Net

From the trained weights of a COVID-Net for COVID-19 identification in radiographs,
this tool performs transfer learning to re-use these weights for stratification of
patient offset (# of days since symptoms began *)

Steps to use this:
1. follow instructions for building data dir with train & test subdirs
2. train your network with the train_tf.py script
3. run this script and pass the path to your trained network (defaults should suffice)

(*) FIXME: It seems that the definition of offset varies between data sources! (account for this)
TODO: Make this script more general so that it can be used to transfer learn for other applications

paul@darwinai.ca
"""
import argparse
from collections import namedtuple
import cv2
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from data import BalanceDataGenerator


# We will create a checkpoint which has initial values for these variables
VARS_TO_FORGET = [
    'dense_3/kernel:0',
    'dense_3/bias:0',
    'dense_2/kernel:0',
    'dense_2/bias:0',
    'dense_1/kernel:0',
    'dense_1/bias:0',
]
IMAGE_SHAPE = (224, 224, 3)
INPUT_TENSOR_NAME = "input_1:0"
OUTPUT_TENSOR_NAME = "dense_3/Softmax:0"
SAMPLE_WEIGHTS = "dense_3_sample_weights:0"


def get_parse_fn(num_classes: int, augment: bool = False):
    def parse_function(imagepath: str, label: int):
        """Parse a single element of the stratification dataset"""
        # TODO add augmentation here ideally
        image_decoded = tf.image.resize_images(
            tf.image.decode_jpeg(tf.io.read_file(imagepath), IMAGE_SHAPE[-1]), IMAGE_SHAPE[:2])
        return (
            tf.image.convert_image_dtype(image_decoded, dtype=tf.float32) / 255.0, # x
            tf.one_hot(label, num_classes), # y
            tf.convert_to_tensor(1.0, dtype=tf.float32), # sample_weights TODO: verify this is right
        )
    return parse_function


def parse_split(split_txt_path: str) -> Tuple[List[str], List[int]]:
    """Read the offsets for COVID patients based on the files in our split"""
    # FIXME: ideally we should just store the offset in the split as well or read it from CSV by id.
    # FIXME: we need to add pretrained weights + .txts for split with well-distributed offset.
    files, labels = [], [],
    for split_entry in open(split_txt_path).readlines():
        _, image_file, diagnosis = split_entry.strip().split() # TODO: txts should just contain ids
        if diagnosis == 'COVID-19':
            patient = csv[csv["filename"] == image_file]
            recorded_offset = patient['offset'].item()
            if not np.isnan(recorded_offset):
                offset = stratify(int(recorded_offset))
                image_path = os.path.abspath(
                    os.path.join(args.chestxraydir, 'images', image_file))
                assert os.path.exists(image_path), "Missing file {}".format(image_path)
                files.append(image_path)
                labels.append(offset)
    return files, labels


def eval_net(sess: tf.Session, dataset_dict: Dict[str, Any], test_files: List[str],
             test_labels: List[int]) -> None:
    """Evaluate the network"""
    # Reset eval iterator
    sess.run(dataset_dict['iterator'].initializer)

    # Eval
    preds, all_labels = [], []
    num_evaled = 0
    while True:
        try:
            images, labels, sample_weights = sess.run(dataset_dict['gn_op'])
            pred = sess.run(
                OUTPUT_TENSOR_NAME,
                feed_dict={INPUT_TENSOR_NAME: images, SAMPLE_WEIGHTS: sample_weights}
            )
            preds.append(np.array(pred).argmax(axis=1))
            num_evaled += len(pred)
            all_labels.extend(np.array(labels).argmax(axis=1))
        except tf.errors.OutOfRangeError:
            print("\tevaluated {} images.".format(num_evaled))
            break

    matrix = confusion_matrix(all_labels, np.concatenate(preds)).astype('float')
    per_class_acc = [
        matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))
    ]
    print("confusion matrix:\n{}\nper-class accuracies:\n{}".format(matrix, per_class_acc))


if __name__ == "__main__":

    # Input args NOTE: the params here differ from thise in train_tf.py - we are fine-tuning
    parser = argparse.ArgumentParser(description='COVIDNet-Risk Transfer Learning Script (offset).')
    parser.add_argument('--classes', default=4, type=int,
                        help='Number of classes to stratify offset into.')
    parser.add_argument('--stratification', type=int, nargs='+', default=[3, 5, 10],
                        help='Stratification points (days), i.e. "5 10" produces stratification of'
                        ': 0o <-0c-> 5o <-1c-> 10o -2c-> via >= comparison (o=offset, c=class).')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs (less since we\'re effectively fine-tuning).')
    parser.add_argument('--lr', default=0.000002, type=float, help='Learning rate.')
    parser.add_argument('--batch-size', default=8, type=int, help='Train batch-size')
    parser.add_argument('--eval-batch-size', default=8, type=int, help='Eval batch-size')
    parser.add_argument('--evaliterval', default=3, type=int,
                        help='# of epochs to train before running evaluation. NOTE: we only save'
                        'after evaluation. This can be disabled when more test data is available')
    parser.add_argument('--input-weights-dir', default='models/COVIDNetv2', type=str,
                        help='Path to input folder containing a trained COVID-Netv2 checkpoint')
    parser.add_argument('--input-meta-name', default='model.meta', type=str,
                        help='Name of meta file within <input-weights-dir>')
    parser.add_argument('--outputdir', default='models/COVIDNet-Risk', type=str,
                        help='Path to output folder.')
    parser.add_argument('--trainfile', default='train_COVIDx.txt', type=str,
                        help='Name of train file. NOTE: stock split is insufficient at this time.')
    parser.add_argument('--testfile', default='test_COVIDx.txt', type=str,
                        help='Name of test file. NOTE: stock split is insufficient at this time.')
    parser.add_argument('--name', default='COVIDNet-Risk', type=str,
                        help='Name of folder to store training checkpoints.')
    parser.add_argument('--chestxraydir', default='../covid-chestxray-dataset', type=str,
                        help='Path to the chestxray images directory for COVID-19 patients.')
    args = parser.parse_args()

    # Check inputs
    assert os.path.exists(args.input_weights_dir), "Missing file {}".format(args.input_weights_dir)
    assert os.path.exists(os.path.join(args.input_weights_dir, args.input_meta_name)), \
        "Missing file {}".format(args.input_meta_name)

    # Format and define a stratification method based on our points
    # TODO we could do a different amount of stratification but we have to add our own dense layers
    assert len(args.stratification) == 3, "Must pass exactly 3 offset stratification points"
    if args.stratification[0] != 0:
        stratification = np.array([0, *args.stratification])
    else:
        stratification = np.array(args.stratification)
    num_classes = len(stratification)
    stratify = lambda offset: np.where(offset >= stratification)[0][-1]

    # Read CSV of dataset
    assert os.path.exists(args.chestxraydir), "please clone "\
        "https://github.com/ieee8023/covid-chestxray-dataset and pass path to dir as --chestxraydir"
    csv = pd.read_csv(os.path.join(args.chestxraydir, "metadata.csv"), nrows=None)

    # Get the image filepaths and labels for training and testing split
    train_files, train_labels = parse_split(args.trainfile)
    assert len(train_files) >= 0 and len(train_files) == len(train_labels)
    test_files, test_labels = parse_split(args.testfile)
    assert len(test_files) >= 0 and len(test_labels) == len(test_files)
    print("collected {} training and {} test cases for transfer-learning".format(
        len(train_files), len(test_files)))

    # Init augmentation fn - FIXME: we need a way to put this in a parse_fn for tf.data.dataset
    # augmentation_fn = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    #     brightness_range=(0.9, 1.1),
    #     fill_mode='constant',
    #     cval=0.,
    # )
    # < define generator from augmentation_fn + cv loads? >
    # dataset = tf.data.Dataset.from_generator(lambda: generator,
    #                                       output_types=(tf.float32, tf.float32, tf.float32),
    #                                       output_shapes=([batch_size, 224, 224, 3],
    #                                                      [batch_size, 3],
    #                                                      [batch_size]))

    # Output path creation for this run with lr param in name
    train_dir = os.path.join(args.outputdir, args.name + '-lr' + str(args.lr))
    os.makedirs(args.outputdir, exist_ok=True)
    os.makedirs(train_dir)
    print('Output: ' + train_dir)

    # Train
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        # Import meta graph
        tf.train.import_meta_graph(os.path.join(args.input_weights_dir, args.input_meta_name))

        # Restore pre-trained vars which are not in our VARS_TO_FORGET list
        restore_vars_list, init_vars_list = [], []
        for var in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if var.name in VARS_TO_FORGET:
                init_vars_list.append(var)
            else:
                restore_vars_list.append(var)
        restore_saver = tf.train.Saver(var_list=restore_vars_list)
        restore_saver.restore(sess, tf.train.latest_checkpoint(args.input_weights_dir))
        existing_vars = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Get some I/O tensors
        image_tensor = graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        labels_tensor = graph.get_tensor_by_name("dense_3_target:0")
        sample_weights = graph.get_tensor_by_name(SAMPLE_WEIGHTS)
        pred_tensor = graph.get_tensor_by_name("dense_3/MatMul:0")

        # Define tf.datasets
        datasets = {}
        for is_training, files, labels in zip(
                [True, False], [train_files, test_files], [train_labels, test_labels]):

            dataset = tf.data.Dataset.from_tensor_slices((files, labels))
            dataset = dataset.map(get_parse_fn(num_classes))
            if is_training:
                dataset = dataset.shuffle(15)
            dataset = dataset.batch(args.batch_size if is_training else args.eval_batch_size)
            if is_training:
                dataset = dataset.repeat()
            iterator = dataset.make_initializable_iterator()
            datasets['train' if is_training else 'test'] = {
                'dataset': dataset,
                'iterator': iterator,
                'gn_op': iterator.get_next(),
            }

        # Define loss and optimizer
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=pred_tensor, labels=labels_tensor) * sample_weights
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op)
        optim_vars = list(
            set(sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(existing_vars))

        # Initialize the optimizer + dsi + vars in our VARS_TO_FORGET list
        sess.run(tf.variables_initializer(optim_vars + init_vars_list))

        # save base model
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(train_dir, 'model'))
        print('Saved pre-trained model with re-initialized output layers.')
        print('Baseline eval:')
        eval_net(sess, datasets['test'], test_files, test_labels)

        # Training cycle
        # TODO: we need a training method that we can re-use. below very similar to train_tf.py
        # FIXME: we need to consider freezing vars for all but dense layers.
        print('Transfer Learning Started.')
        print('\ttrain samples: {}\n\ttest samples:  {}\n\tstratification: {}\n'.format(
            len(train_files), len(test_files), args.stratification))
        sess.run(datasets['train']['iterator'].initializer)
        num_batches = len(train_files) // args.batch_size
        progbar = tf.keras.utils.Progbar(num_batches)
        for epoch in range(args.epochs):

            # Train
            print("Fine-Tuning on 1 epoch = {} images.".format(len(train_files)))
            for i in range(num_batches):

                batch_x, batch_y, weights = sess.run(datasets['train']['gn_op'])
                sess.run(
                    train_op,
                    feed_dict={
                        image_tensor: batch_x,
                        labels_tensor: batch_y,
                        sample_weights: weights,
                    }
                )
                progbar.update(i + 1)

            # Evaluate + save
            if epoch % args.evaliterval == 0:
                pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                loss = sess.run(
                    loss_op,
                    feed_dict={
                        pred_tensor: pred,
                        labels_tensor: batch_y,
                        sample_weights: weights,
                    }
               )
                print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
                eval_net(sess, datasets['test'], test_files, test_labels)
                saver.save(
                    sess,
                    os.path.join(train_dir, 'model'),
                    global_step=epoch + 1,
                    write_meta_graph=False
                )
                print('Saving checkpoint at epoch {}'.format(epoch + 1))

    print("Transfer Learning Finished!\n\tcheckpoint: '{}'".format(train_dir))
