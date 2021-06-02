from __future__ import print_function
import tensorflow as tf
import os

import argparse
import pathlib
import datetime
import numpy as np  # for debugging
from tensorflow.keras import backend as K

from eval_tf_external import eval
from data_tf import COVIDxDataset
from model import build_UNet2D_4L, build_resnet_attn_model
from load_data import loadDataJSRTSingle
from utils.tensorboard import heatmap_overlay_summary_op, scalar_summary,log_tensorboard_images

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--col_name', nargs='+', default=["folder_name", "img_path", "class"])
parser.add_argument('--target_name', type=str, default="class")
parser.add_argument('--weightspath', default='/home/hossein.aboutalebi/data/urgent_sev/0.85', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='2021-05-21#18-16-44.464148COVIDNet-lr8e-05_27',
                    type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/train_pnemunia.txt', type=str, help='Path to train file')
parser.add_argument('--cuda_n', type=str, default="0", help='cuda number')
parser.add_argument('--testfile', default='labels/test_pnemunia.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/hossein.aboutalebi/data/pneumonia/images/', type=str,
                    help='Path to data folder')
parser.add_argument('--in_sem', default=0, type=int,
                    help='initial_itrs until training semantic')
parser.add_argument('--covid_weight', default=1, type=float, help='Class weighting for covid')
parser.add_argument('--covid_percent', default=0.5, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_2/Softmax:0', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logged_images', default='labels/logged_p.txt', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str,
                    help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str,
                    help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str,
                    help='Name of sample weights tensor for loss')
parser.add_argument('--training_tensorname', default='keras_learning_phase:0', type=str,
                    help='Name of training placeholder tensor')


height_semantic = 256  # do not change unless train a new semantic model
width_semantic = 256
switcher = 3

args = parser.parse_args()

# Parameters
learning_rate = args.lr
batch_size = args.bs
test_batch_size = 50
display_step = 1    # evaluation interval in epochs
log_interval = 100  # image and loss log interval in steps (batches)
class_weights = [1., args.covid_weight]

# Make output paths
current_time = (str(datetime.datetime.now()).replace(" ", "#")).replace(":", "-")
outputPath = './output/' + current_time
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)

print('Output: ' + runPath)

# Load list of test files
with open(args.testfile) as f:
    testfiles = f.readlines()

dataset = COVIDxDataset(
    args.datadir, num_classes=2, image_size=args.input_size,
    sem_image_size=width_semantic, class_weights=class_weights)

with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.out_tensorname)
    logit_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    is_training = graph.get_tensor_by_name(args.training_tensorname)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Initialize update ops collection
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('Number of update ops: ', len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

    # Create train ops
    with tf.control_dependencies(extra_ops):
        train_op = optimizer.minimize(loss_op)

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Load weights
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    # Save base model and run baseline eval
    saver = tf.train.Saver(max_to_keep=100)
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    metrics = eval(
        sess, dataset, args.testfile, test_batch_size, image_tensor,
        pred_tensor, dataset.class_map)

    # Training cycle
    print('Training started')
    train_dataset, count, batch_size = dataset.train_dataset(args.trainfile, batch_size)
    data_next = train_dataset.make_one_shot_iterator().get_next()
    total_batch = int(np.ceil(count/batch_size))
    progbar = tf.keras.utils.Progbar(total_batch)

    for epoch in range(args.epochs):
        for i in range(total_batch):
            # Get batch of data
            data = sess.run(data_next)
            batch_x = data['image']
            batch_sem_x = data['sem_image']
            batch_y = data['label']
            weights = data['weight']
            feed_dict = {
                image_tensor: batch_x,
                labels_tensor: batch_y,
                sample_weights: weights,
                is_training: 1}

            sess.run(train_op, feed_dict=feed_dict)

            total_steps = epoch*total_batch + i
            progbar.update(i + 1)

        if epoch % display_step == 0:
            # Print minibatch loss and lr
            loss = sess.run(loss_op, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            print("lr: {},  batch_size: {}".format(str(args.lr),str(args.bs)))

            # Run eval and log results to tensorboard
            metrics = eval(sess, dataset, args.testfile, test_batch_size, image_tensor,
                           pred_tensor, dataset.class_map)
            print('Output: ' + runPath+"_"+str(epoch))
            print('Saving checkpoint at epoch {}'.format(epoch + 1))
            saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)

print("Optimization Finished!")
