from __future__ import print_function
import tensorflow as tf
import os

import argparse
import pathlib
import datetime
import numpy as np  # for debugging
from tensorflow.keras import backend as K

from eval import eval
from data import BalanceCovidDataset, process_image_file
from model import build_UNet2D_4L, build_resnet_attn_model
from load_data import loadDataJSRTSingle
from utils.tensorboard import heatmap_overlay_summary_op, scalar_summary,log_tensorboard_images

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_keras_collections(graph, keras_model):
    """
    Creates missing collections in a tf.Graph using keras model attributes
    Args:
        graph (tf.Graph): Tensorflow graph with missing collections
        keras_model (keras.Model): Keras model with desired attributes
    """
    if hasattr(keras_model, 'metrics'):
        for metric in keras_model.metrics:
            for update_op in metric.updates:
                graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
            for weight in metric._non_trainable_weights:
                graph.add_to_collection(tf.GraphKeys.METRIC_VARIABLES, weight)
                graph.add_to_collection(tf.GraphKeys.LOCAL_VARIABLES, weight)
    else:
        print('skipped adding variables from metrics')

    for update_op in keras_model.updates:
        graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    # Clear default trainable collection before adding tensors
    graph.clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for trainable_layer in keras_model.trainable_weights:
        graph.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, trainable_layer)


parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--col_name', nargs='+', default=["folder_name", "img_path", "class"])
parser.add_argument('--target_name', type=str, default="class")
parser.add_argument('--weightspath', default='/home/hossein.aboutalebi/data/sem/0.983', type=str, help='Path to output folder')
# parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='2021-04-30#15-31-59.224643COVIDNet-lr8e-05_22',
                    type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/train_COVIDx8B.txt', type=str, help='Path to train file')
parser.add_argument('--cuda_n', type=str, default="0", help='cuda number')
parser.add_argument('--testfile', default='labels/test_COVIDx8B.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/maya.pavlova/covidnet-orig/data', type=str,
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
parser.add_argument('--logged_images', default='labels/logged_images.txt', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str,
                    help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str,
                    help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str,
                    help='Name of sample weights tensor for loss')
parser.add_argument('--load_weight', action='store_true',
                    help='default False')
parser.add_argument('--resnet_type', default='resnet1', type=str,
                    help='type of resnet arch. Values can be: resnet0_M, resnet0_R, resnet1, resnet2')
parser.add_argument('--training_tensorname', default='keras_learning_phase:0', type=str,
                    help='Name of training placeholder tensor')


height_semantic = 256  # do not change unless train a new semantic model
width_semantic = 256
switcher = 3

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_n

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1    # evaluation interval in epochs
log_interval = 100  # image and loss log interval in steps (batches)

# Make output paths
current_time = (str(datetime.datetime.now()).replace(" ", "#")).replace(":", "-")
outputPath = './output/' + current_time
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
# path_images_train=os.path.join(runPath,"images/train")
# path_images_test=os.path.join(runPath,"images/test")
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
# pathlib.Path(path_images_train).mkdir(parents=True, exist_ok=True)
# pathlib.Path(path_images_test).mkdir(parents=True, exist_ok=True)

print('Output: ' + runPath)

# Load list of test files
# testfiles_frame = pd.read_csv(args.testfile, delimiter=" ", names=args.col_name).values
with open(args.testfile) as f:
    testfiles = f.readlines()

# Get image file names to log throughout training
with open(args.logged_images) as f:
    log_images = f.readlines()

# Get stack of images to log
log_positive, log_negative = [], []
for i in range(len(log_images)):
    line = log_images[i].split()
    # image = process_image_file(os.path.join(args.datadir, 'test', line[1]), 0.08, args.input_size)
    # image = image.astype('float32') / 255.0
    sem_image = loadDataJSRTSingle(os.path.join(args.datadir, 'test', line[1]), (width_semantic, width_semantic))
    if line[2] == 'positive':
        log_positive.append(sem_image)
    elif line[2] == 'negative':
        log_negative.append(sem_image)
log_positive, log_negative = np.array(log_positive), np.array(log_negative)

generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=batch_size,
                                input_shape=(args.input_size, args.input_size),
                                covid_percent=args.covid_percent,
                                class_weights=[1., 1.],
                                top_percent=args.top_percent,
                                col_name=args.col_name,
                                target_name=args.target_name,
                                semantic_input_shape=(width_semantic, width_semantic))

with tf.Session() as sess:
    K.set_session(sess)
    # First we load the semantic model:
    model_semantic = build_UNet2D_4L((height_semantic, width_semantic, 1))
    labels_tensor = tf.placeholder(tf.float32)
    sample_weights = tf.placeholder(tf.float32)

    batch_x, batch_sem_x, batch_y, weights = next(generator)
    resnet_50 = build_resnet_attn_model(name=args.resnet_type, classes=2, model_semantic=model_semantic)
    training_ph = K.learning_phase()
    model_main = resnet_50.call(input_shape=(args.input_size, args.input_size, 3), training=training_ph)

    image_tensor = model_main.input[0]  # The model.input is a tuple of (input_2:0, and input_1:0)
    semantic_image_tensor = model_semantic.input
    print(image_tensor.name)
    print(labels_tensor.name)
    print(semantic_image_tensor.name)
    print(model_main.output.name)
    print(sample_weights.name)

    graph = tf.get_default_graph()
    pred_tensor = model_main.output
    saver = tf.train.Saver(max_to_keep=100)

    logit_tensor = graph.get_tensor_by_name('final_output/MatMul:0')

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Initialize update ops collection
    init_keras_collections(graph, model_main)
    print('length with model_main: ', len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
    # init_keras_collections(graph, model_semantic)
    # print('length with model_semantic: ', len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

    # Create train ops
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_vars_resnet = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "^((?!sem).)*$")
    train_vars_sem = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sem*")
    with tf.control_dependencies(extra_ops):
        train_op_all = optimizer.minimize(loss_op, var_list=train_vars)
        train_op_resnet = optimizer.minimize(loss_op, var_list=train_vars_resnet)
        if args.resnet_type[:7] != 'resnet0':
            train_op_sem = optimizer.minimize(loss_op, var_list=train_vars_sem)
        print('Train vars resnet: ', len(train_vars_resnet))
        print('Train vars semantic: ', len(train_vars_sem))

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Make summary ops and writer
    loss_summary = tf.summary.scalar('train/loss', loss_op)
    image_summary = heatmap_overlay_summary_op(
        'train/semantic', model_semantic.input, model_semantic.output, max_outputs=5)
    test_image_summary_pos = heatmap_overlay_summary_op(
        'test/semantic/positive', model_semantic.input, model_semantic.output, max_outputs=len(log_images))
    test_image_summary_neg = heatmap_overlay_summary_op(
        'test/semantic/negative', model_semantic.input, model_semantic.output, max_outputs=len(log_images))
    summary_op = tf.summary.merge([loss_summary, image_summary])
    summary_writer = tf.summary.FileWriter(os.path.join(runPath, 'events'), graph)

    # Load weights
    if args.load_weight:
        saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    else:
        model_semantic.load_weights("./model/trained_model.hdf5")
    # saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

    # Save base model and run baseline eval
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')
    summary_pos, summary_neg = log_tensorboard_images(sess, K,test_image_summary_pos, semantic_image_tensor, log_positive,
                                                      test_image_summary_neg, log_negative)
    summary_writer.add_summary(summary_pos, 0)
    summary_writer.add_summary(summary_neg, 0)
    print("Finished tensorboard baseline")
    metrics = eval(
        sess, model_semantic, testfiles, os.path.join(args.datadir, 'test'), image_tensor, semantic_image_tensor,
        pred_tensor, args.input_size, width_semantic, batch_size=batch_size, mapping=generator.mapping)
    summary_writer.add_summary(scalar_summary(metrics, 'val/'), 0)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)

    for epoch in range(args.epochs):
        # Select train op depending on training stage
        if epoch < args.in_sem or epoch % switcher != 0 or args.resnet_type[:7] == 'resnet0':
            train_op = train_op_resnet
        else:
            train_op = train_op_sem

        # Log images and semantic output
        summary_pos,summary_neg=log_tensorboard_images(sess,K,test_image_summary_pos,semantic_image_tensor,log_positive,test_image_summary_neg,log_negative)
        summary_writer.add_summary(summary_pos, epoch)
        summary_writer.add_summary(summary_neg, epoch)

        for i in range(total_batch):
            batch_x, batch_sem_x, batch_y, weights = next(generator)
            total_steps = epoch*total_batch + i
            if not (total_steps % log_interval):  # run summary op for batch
                _, pred, semantic_output, summary = sess.run(
                    (train_op, pred_tensor, model_semantic.output, summary_op),
                    feed_dict={image_tensor: batch_x,
                               semantic_image_tensor: batch_sem_x,
                               labels_tensor: batch_y,
                               sample_weights: weights,
                               K.learning_phase(): 1})
                summary_writer.add_summary(summary, total_steps)
            else:  # run without summary op
                _, pred, semantic_output = sess.run((train_op, pred_tensor, model_semantic.output),
                                                    feed_dict={image_tensor: batch_x,
                                                                semantic_image_tensor: batch_sem_x,
                                                                labels_tensor: batch_y,
                                                                sample_weights: weights,
                                                                K.learning_phase(): 1})
            progbar.update(i + 1)

        if epoch % display_step == 0:
            # Print minibatch loss and lr
            # semantic_output=model_semantic(batch_x.astype('float32')).eval(session=sess)
            # pred = model_main((batch_x.astype('float32'),semantic_output)).eval(session=sess)
            loss = sess.run(loss_op, feed_dict={image_tensor: batch_x,
                                          semantic_image_tensor: batch_sem_x,
                                          labels_tensor: batch_y,
                                          sample_weights: weights,
                                          K.learning_phase(): 1})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            print("lr: {},  batch_size: {}".format(str(args.lr),str(args.bs)))

            # Run eval and log results to tensorboard
            metrics = eval(
                sess, model_semantic, testfiles, os.path.join(args.datadir, 'test'), image_tensor,
                semantic_image_tensor, pred_tensor, args.input_size, width_semantic, mapping=generator.mapping)
            summary_writer.add_summary(scalar_summary(metrics, 'val/'), (epoch + 1)*total_batch)
            model_main.save_weights(runPath+"_"+str(epoch))
            print('Output: ' + runPath+"_"+str(epoch))
            print('Saving checkpoint at epoch {}'.format(epoch + 1))

print("Optimization Finished!")
