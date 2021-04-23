from __future__ import print_function
import pandas as pd
import tensorflow as tf
import os, argparse, pathlib
import datetime
import numpy as np #for debugging
from tensorflow.keras import backend as K

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model.resnet import ResnetBuilder
from model.resnet2 import ResNet50
from eval import eval
from data import BalanceCovidDataset
from model.build_model import build_UNet2D_4L

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=40, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--col_name', nargs='+', default=["folder_name", "img_path", "class"])
parser.add_argument('--target_name', type=str, default="class")
parser.add_argument('--weightspath', default='output/sev_models/covidnet-cxr-2',
                    type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1705', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/train_COVIDx8B.txt', type=str, help='Path to train file')
parser.add_argument('--cuda_n', type=str, default="0", help='cuda number')
parser.add_argument('--testfile', default='labels/test_COVIDx8B.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/maya.pavlova/covidnet-orig/data', type=str,
                    help='Path to data folder')
parser.add_argument('--in_sem', default=200, type=int,
                    help='initial_itrs until training semantic')
parser.add_argument('--covid_weight', default=1, type=float, help='Class weighting for covid')
parser.add_argument('--covid_percent', default=0.5, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_2/Softmax:0', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str,
                    help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str,
                    help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str,
                    help='Name of sample weights tensor for loss')
parser.add_argument('--load_weight', action='store_true',
                    help='default False')
parser.add_argument('--training_tensorname', default='keras_learning_phase:0', type=str,
                    help='Name of training placeholder tensor')

height_semantic = 256  # do not change unless train a new semantic model
width_semantic = 256
switcher=3

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_n

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1

# output path
current_time = (str(datetime.datetime.now()).replace(" ", "#")).replace(":", "-")
outputPath = './output/' + current_time
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# testfiles_frame = pd.read_csv(args.testfile, delimiter=" ", names=args.col_name).values
with open(args.testfile) as f:
    testfiles = f.readlines()

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
    model_semantic.load_weights("./model/trained_model.hdf5")
    labels_tensor =  tf.placeholder(tf.float32)
    sample_weights = tf.placeholder(tf.float32)

    batch_x, batch_sem_x, batch_y, weights = next(generator)
    # model_main = ResnetBuilder.build_resnet_50(input_shape=( args.input_size, args.input_size,3),
    #                                            width_semantic=width_semantic, num_outputs=2,
    #                                            model_semantic=model_semantic)
    resnet_50=ResNet50(classes=2, model_semantic=model_semantic)
    model_main=resnet_50.call(input_shape=(args.input_size, args.input_size, 3))

    # print('semantic model output: ', model_semantic.output)
    image_tensor = model_main.input[0] # The model.input is a tuple of (input_2:0, and input_1:0)
    semantic_image_tensor = model_semantic.input
    # pred_tensor = model_main(batch_x)

    graph = tf.get_default_graph()
    pred_tensor=model_main.output
    saver = tf.train.Saver(max_to_keep=100)

    logit_tensor = graph.get_tensor_by_name('final_output/MatMul:0')


    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_vars_resnet = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "^((?!sem).)*$")
    # print(train_vars_resnet)
    train_vars_sem = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sem*")
    train_op_resnet = optimizer.minimize(loss_op, var_list=train_vars_resnet)
    train_op_sem = optimizer.minimize(loss_op, var_list=train_vars_sem)
    # print('All train vars: ', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    print('Train vars resnet: ', len(train_vars_resnet))
    print('Train vars semantic: ', len(train_vars_sem))

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    if (args.load_weight):
        saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    # saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

    # save base model
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')
    # eval(sess, graph, testfiles, os.path.join(args.datadir, 'test'),
    #      image_tensor, semantic_image_tensor, pred_tensor, args.input_size, width_semantic, mapping=generator.mapping)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)
    for epoch in range(args.epochs):
        if (epoch < args.in_sem or epoch % switcher != 0):
            train_op = train_op_resnet 
        else:
            train_op = train_op_sem
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_sem_x, batch_y, weights = next(generator)
            # print('labels: ', batch_y)
            # print('weights: ', weights)
            _, pred, semantic_output = sess.run((train_op, pred_tensor, model_semantic.output),
                                          feed_dict={image_tensor: batch_x,
                                          semantic_image_tensor: batch_sem_x,
                                          model_semantic.output: batch_sem_x,
                                          labels_tensor: batch_y,
                                          sample_weights: weights,
                                          K.learning_phase(): 1})
            # print('semantic results:')
            # print(semantic_output)
            # print('pred results')
            # print(pred)
            progbar.update(i + 1)

        if epoch % display_step == 0:
            # semantic_output=model_semantic(batch_x.astype('float32')).eval(session=sess)
            # pred = model_main((batch_x.astype('float32'),semantic_output)).eval(session=sess)
            loss = sess.run(loss_op, feed_dict={image_tensor: batch_x,
                                          semantic_image_tensor: batch_sem_x,
                                          model_semantic.output: batch_sem_x,
                                          labels_tensor: batch_y,
                                          sample_weights: weights,
                                          K.learning_phase(): 1})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            print("lr: {},  batch_size: {}".format(str(args.lr),str(args.bs)))
            eval(sess, model_semantic, testfiles, os.path.join(args.datadir, 'test'),
                 image_tensor, semantic_image_tensor, pred_tensor, args.input_size, width_semantic, mapping=generator.mapping)
            # saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch + 1, write_meta_graph=False)
            model_main.save_weights(runPath+"_"+str(epoch))
            print('Output: ' + runPath+"_"+str(epoch))
            print('Saving checkpoint at epoch {}'.format(epoch + 1))

print("Optimization Finished!")
