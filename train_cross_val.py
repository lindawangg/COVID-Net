from __future__ import print_function
from random import sample
import random
import tensorflow as tf
import numpy as np
import os, argparse, pathlib
import logging
from datetime import datetime

from eval import eval
from data_cross_val import BalanceCovidDataset
from sklearn.model_selection import KFold, StratifiedKFold

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return np.array(files)

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='/home/maya.pavlova/covidnet-orig/models/COVIDNet-CXR-2', type=str, help='Path to model files, defaults to \'models/COVIDNet-CXR-2\'')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes, defaults to 2')
parser.add_argument('--file', default='/home/maya.pavlova/covidnet-orig/hospital_data.txt', type=str, help='Path to train file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/maya.pavlova/covidnet-orig/hospital_images', type=str, help='Path to data folder')
parser.add_argument('--covid_weight', default=1., type=float, help='Class weighting for covid')
parser.add_argument('--covid_percent', default=0.3, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_2/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str, help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str, help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str, help='Name of sample weights tensor for loss')
parser.add_argument('--cuda_n', type=str, default="0", help='cuda number')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_n

# Set up output paths
outputPath = './output/'
runID = args.name + '-cross_val-' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# Set up logger
logging.basicConfig(filename=os.path.join(runPath, 'performance.log'), filemode='w')

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1
n_MCarlo=10
if args.n_classes == 2:
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

# Set up folds
fold_number = 5
fold_repetition = 50
files = list(_process_csv_file(args.file))
classes=[element.split(" ")[-1][:-1] for element in files]


print("creating balanced negative percentage")
list_negative=[]
preserved_neg=[]
for i in range(len(classes)):
    if(classes[i]=="negative"):
        list_negative.append(i)
print(list_negative)
list_negative.sort(reverse=True)
for index in list_negative:
    preserved_neg.append(files[index])
    del classes[index]
    del files[index]
random.shuffle(list_negative)
random.shuffle(preserved_neg)
step_size=int(np.floor(len(preserved_neg)/fold_number))
chunks_neg = [preserved_neg[x:x + step_size] for x in range(0, len(preserved_neg), step_size)]
if(len(chunks_neg)>fold_number):
    chunks_neg[-1]= chunks_neg[-1] + chunks_neg[-2]
    del chunks_neg[-2]

print('length of all files (which should be just positive here): ', len(files))
print('what chunks_neg is each fold')
for i in range(5):
    print('fold ', i)
    print(' test files len: ', len(chunks_neg[i]))
    print(chunks_neg[i])
    temp = []
    for fold in range(fold_number):
        if fold != i:
            temp += chunks_neg[fold]
    print('train files len: ', len(temp))
    print(temp)
    print()

kf = KFold(n_splits=fold_number, random_state=42, shuffle=True)

with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver = tf.train.Saver(max_to_keep=100)

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    for fold_num, (train_i, test_i) in enumerate(kf.split(files)):
        logging.info('FOLD NUMBER: {}'.format(fold_num))
        min_loss, max_acc = float('inf'), 0

        for fold_rep in range(fold_repetition):
            logging.info('Repetition number: {}'.format(fold_rep))

            # output path
            fold_run_path = runPath + '-file_num-' + str(fold_num) + 'rep-' + str(fold_rep)
            pathlib.Path(fold_run_path).mkdir(parents=True, exist_ok=True)
            print('Output fold number file path: ' + runPath)

            # For K-fold normal:
            # trainfiles = np.array(files)[train_i]
            # print('len of ')
            # To get train files for negative, concatenate all other folds together
            # trainfiles_neg = []
            # for j in range(fold_number):
            #     if j != fold_num:
            #         trainfiles_neg += chunks_neg[j]
            # print('Length of negative training files: {} and test files {}'.format(len(trainfiles_neg), len(chunks_neg[fold_num])))
            # testfiles = np.concatenate((np.array(files)[test_i],np.array(chunks_neg[fold_num])))

            #For Monte Carlo:
            test_neg_MCarlo=sample(preserved_neg,n_MCarlo)
            test_pos_McCarlo=sample(files,n_MCarlo)
            testfiles= np.concatenate((np.array(test_neg_MCarlo),np.array(test_pos_McCarlo)))
            trainfiles=np.array([element for element in files if element not in test_pos_McCarlo])
            trainfiles_neg=[element for element in preserved_neg if element not in test_neg_MCarlo]

            generator = BalanceCovidDataset(data_dir=args.datadir,
                                            files=trainfiles,
                                            neg_files=trainfiles_neg,
                                            batch_size=batch_size,
                                            input_shape=(args.input_size, args.input_size),
                                            n_classes=args.n_classes,
                                            mapping=mapping,
                                            covid_percent=args.covid_percent,
                                            class_weights=class_weights,
                                            top_percent=args.top_percent)

            # Initialize the variables
            init = tf.global_variables_initializer()

            # Run the initializer for every new k-fold run
            sess.run(init)

            # load weights
            saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
            #saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

            # save base model
            saver.save(sess, os.path.join(fold_run_path, 'model'))
            print('Saved baseline checkpoint')
            print('Baseline eval:')
            acc, class_acc, ppvs = eval(sess, graph, testfiles, args.datadir,
                args.in_tensorname, args.out_tensorname, args.input_size, mapping)
            if acc > max_acc:
                logging.info('fold number: {}, repetition: {}, epoch: {}'.format(fold_num, fold_rep, 'baseline'))
                logging.info('accuracy: {}'.format(acc))
                logging.info('Sens' + ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
                logging.info('PPV' + ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))
                max_acc = acc

            # Training cycle
            print('Training started')
            total_batch = len(generator)
            progbar = tf.keras.utils.Progbar(total_batch)
            for epoch in range(args.epochs):
                for i in range(total_batch):
                    # Run optimization
                    batch_x, batch_y, weights = next(generator)
                    sess.run(train_op, feed_dict={image_tensor: batch_x,
                                                labels_tensor: batch_y,
                                                sample_weights: weights})
                    progbar.update(i+1)

                if epoch % display_step == 0:
                    pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                    loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                        labels_tensor: batch_y,
                                                        sample_weights: weights})
                    acc, class_acc, ppvs = eval(sess, graph, testfiles, args.datadir,
                        args.in_tensorname, args.out_tensorname, args.input_size, mapping)
                    if loss < min_loss or acc > max_acc:
                        logging.info('fold number: {}, repetition: {}, epoch: {}, minibatch loss: {}'.format(fold_num, fold_rep, epoch + 1, loss))
                        logging.info('fold number: {}, repetition: {}, epoch: {}'.format(fold_num, fold_rep, epoch + 1))
                        logging.info('accuracy: {}'.format(acc))
                        logging.info('Sens' + ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
                        logging.info('PPV' + ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))
                        max_acc = max(acc, max_acc)
                        min_loss = min(loss, min_loss)

                    saver.save(sess, os.path.join(fold_run_path, 'model'), global_step=epoch+1, write_meta_graph=False)
                    print('Saving checkpoint at epoch {}'.format(epoch + 1))


print("Optimization Finished!")
