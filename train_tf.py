from __future__ import print_function
import tensorflow as tf
import os, argparse, pathlib

from eval import eval
from data import BalanceCovidDataset

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR-2', type=str,
                    help='Path to model files, defaults to \'models/COVIDNet-CXR-2\'')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes, defaults to 2')
parser.add_argument('--trainfile', default='labels/train_COVIDx9B.txt', type=str, help='Path to train file')
parser.add_argument('--testfile', default='labels/test_COVIDx9B.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='data', type=str, help='Path to data folder')
parser.add_argument('--covid_weight', default=1., type=float, help='Class weighting for covid')
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
parser.add_argument('--training_tensorname', default='keras_learning_phase:0', type=str,
                    help='Name of training placeholder tensor')
parser.add_argument('--is_severity_model', action='store_true', 
                    help='Add flag if training COVIDNet CXR-S model')

args = parser.parse_args()

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1

# output path
outputPath = './output/'
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

with open(args.trainfile) as f:
    trainfiles = f.readlines()
with open(args.testfile) as f:
    testfiles = f.readlines()

if args.is_severity_model:
    # For COVIDNet CXR-S severity level 1 and 2 detection using COVIDxSev dataset
    mapping = {
        'level2': 0,
        'level1': 1 
    }
    # For COVIDxSev use a 50/50 balanced batch with 1:1 sample weights
    class_weights = [1., 1.]
    args.covid_percent = 0.5  
elif args.n_classes == 2:
    # For COVID-19 positive/negative detection
    mapping = {
        'negative': 0,
        'positive': 1,
    }
    class_weights = [1., args.covid_weight]
elif args.n_classes == 3:
    # For detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
    mapping = {
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    class_weights = [1., 1., args.covid_weight]
else:
    raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection
        or 3 class detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia''')

generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=batch_size,
                                input_shape=(args.input_size, args.input_size),
                                n_classes=args.n_classes,
                                mapping=mapping,
                                covid_percent=args.covid_percent,
                                class_weights=class_weights,
                                top_percent=args.top_percent,
                                is_severity_model=args.is_severity_model)

with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    training_tensor = graph.get_tensor_by_name(args.training_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    #saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

    # save base model
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')
    eval(sess, graph, testfiles, os.path.join(args.datadir,'test'),
         args.in_tensorname, args.out_tensorname, args.input_size, mapping)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)
    for epoch in range(args.epochs):
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_y, weights, is_training = next(generator)
            sess.run(train_op, feed_dict={image_tensor: batch_x,
                                          labels_tensor: batch_y,
                                          sample_weights: weights,
                                          training_tensor: is_training})
            progbar.update(i+1)

        if epoch % display_step == 0:
            pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
            loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                labels_tensor: batch_y,
                                                sample_weights: weights})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            eval(sess, graph, testfiles, os.path.join(args.datadir,'test'),
                 args.in_tensorname, args.out_tensorname, args.input_size, mapping)
            saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))


print("Optimization Finished!")
