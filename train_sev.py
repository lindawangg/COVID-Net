from __future__ import print_function

import tensorflow as tf
import os, argparse, pathlib

from eval import eval, eval_severity
from data import BalanceCovidDataset

'''
Internal script made for training Montefiore Severity data
Created Oct 4, 2021
'''

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR-2', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='train_mntf_sev.txt', type=str, help='Name of train file')
parser.add_argument('--testfile', default='valid_mntf_sev.txt', type=str, help='Name of test file')
parser.add_argument('--name', default='COVIDNet-MNTF-Sev', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='../montefiore_severity/CXR', type=str, help='Path to data folder')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_2/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str, help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str, help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str, help='Name of sample weights tensor for loss')
parser.add_argument('--sev_reg', action='store_true', default=False, help='Set model to Severity Regression head')
parser.add_argument('--sev_wa', action='store_true', default=False, help='Set model to Severity Weighted Averaging Softmax head')
parser.add_argument('--geo', action='store_true', default=False)
parser.add_argument('--opc', action='store_true', default=False)
parser.add_argument('--gpus', type=int, nargs='*', help='List GPU numbers to use while training')
parser.add_argument('--mae', action='store_true', default=False, help='Use Mean Absolute Error instead of MSE')
parser.add_argument('--msle', action='store_true', default=False, help='Use Mean Squared Log Error')
parser.add_argument('--huber', action='store_true', default=False, help='Use Huber Loss')

args = parser.parse_args()

if not args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    # args.gpus should be list of gpu numbers, e.g. [0, 1, 5, 8]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpus])

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1

# build up output path
outputPath = './output/'
measure = 'geo' if args.geo else 'opc' if args.opc else 'invalid'
if measure == 'invalid': raise ValueError
runID = args.name + '-lr' + str(learning_rate) + f'-{measure}'
if args.mae:
    runID = runID + '-mae'
elif args.msle:
    runID = runID + '-msle'
elif args.huber:
    runID = runID + '-huber'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

with open(args.trainfile) as f:
    trainfiles = f.readlines()
with open(args.testfile) as f:
    testfiles = f.readlines()

generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=batch_size,
                                input_shape=(args.input_size, args.input_size),
                                is_regression=True) # process regr values

# utils for getting dependencies of tensors
def parents(op):
    return set(ip.op for ip in op.inputs)

def children(op):
    return set(op for out in op.outputs for op in out.consumers())

def get_graph():
    ops = tf.get_default_graph().get_operations()
    return {op:children(op) for op in ops}

def print_tf_graph(graph):
    print('PRINTING GRAPH')
    for node in graph:
        for child in graph[node]:
            if 'train' in node.name:
                print("%s -> %s" % (node.name, child.name))

def print_node_parents(node):
    print(f'PRINTING PARENTS OF: {node.name}')
    for p in parents(node):
        print(f'{node.name} -> {p.name}')

def print_node_children(node):
    print(f'PRINTING CHILDREN OF: {node.name}')
    for c in children(node):
        print(f'{node.name} -> {c.name}')

with tf.Session() as sess:
    tf.get_default_graph()
    saver_old = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    # labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    labels_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,1], name='norm_dense_1_target')
    # output of base COVIDNet model (just before prediction head)
    prev_tensor = graph.get_tensor_by_name('flatten_1/Reshape:0')
    prev_tensor.set_shape([None, 460800])

    if args.sev_reg:
        # add another hidden linear layer for kicks
        regr_hl = tf.layers.Dense(64, activation='relu', trainable=True,
                                  name='regr_hl')(prev_tensor)
        # sigmoid activation to force output in (0, 1) range
        regr_head = tf.layers.Dense(1, activation='sigmoid', trainable=True,
                                    name='regr_head')(regr_hl)
        # experimenting with loss functions (though mse seem to work best)
        if args.mae:
            loss_op = tf.reduce_mean(tf.losses.absolute_difference(
                labels=labels_ph, predictions=regr_head))
        elif args.msle:
            loss_op = tf.reduce_mean(tf.math.square(tf.losses.log_loss(
                labels=labels_ph, predictions=regr_head)))
        elif args.huber:
            loss_op = tf.reduce_mean(tf.losses.huber_loss(
                labels=labels_ph, predictions=regr_head, delta=0.1))
        else: # default to MSE
            loss_op = tf.reduce_mean(tf.losses.mean_squared_error(
                labels=labels_ph, predictions=regr_head))
    elif args.sev_wa: # weighted averaging
        # TODO this is where the weighted averaging severity method should go
        raise NotImplementedError
    else: # just leaving here so we know where it fits when this goes back to train_tf.py
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred_tensor, labels=labels_tensor)*sample_weights)

    # Define loss and optimizer
    base_lr = learning_rate
    # use cosine decayed learning rate with warm restarting
    global_step = tf.train.get_or_create_global_step()
    decayed_lr = tf.compat.v1.train.cosine_decay(base_lr, global_step,
                                                 decay_steps=500, alpha=0.01)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver_old.restore(sess, os.path.join(args.weightspath, args.ckptname))

    # save base model
    # build new saver for newly built model architecture
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')

    # found exact string by using:
    #   print_node_children(regr_head.op)
    out_tensorname = 'regr_head/Sigmoid:0'
    eval_severity(sess, graph, testfiles, os.path.join(args.datadir,'valid'),
                  args.in_tensorname, out_tensorname, args.input_size, measure=measure)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)
    for epoch in range(args.epochs):
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_y, weights = next(generator)
            # geo: batch_y[0], opc: batch_y[1], both: batch_y[:]
            if args.geo:
                batch_y = batch_y[:, 0]
            elif args.opc:
                batch_y = batch_y[:, 1]
            else:
                raise ValueError
            # set to size (batch_size, 1) and scale to (0, 1)
            batch_y = batch_y.reshape(batch_size, 1) / 8.0
            sess.run(train_op, feed_dict={image_tensor: batch_x,
                                          labels_ph: batch_y,
                                          sample_weights: weights})
            progbar.update(i+1)

        if epoch % display_step == 0:
            # pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
            loss = sess.run(loss_op, feed_dict={image_tensor:batch_x,
                                                labels_ph: batch_y,
                                                sample_weights: weights})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            eval_severity(sess, graph, testfiles, os.path.join(args.datadir,'valid'),
                          args.in_tensorname, out_tensorname, args.input_size, measure=measure)
            saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=True)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))


print("Optimization Finished!")
