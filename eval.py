from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file
from load_data import loadDataJSRTSingle


def fix_input_image(path_image,input_size,width_semantic,top_percent=0.08,num_channel=3):
    x=np.zeros(( 2,input_size,input_size,num_channel))
    x[0] = process_image_file(path_image, top_percent, input_size)
    x[0] = x[0] / 255.0
    x[1][:width_semantic,:width_semantic,:1]=loadDataJSRTSingle(path_image,
                                   (width_semantic,width_semantic))
    return x.astype('float32')

def eval(sess, graph, testfile, testfolder, input_tensor, input_semantic_tensor, pred_tensor, input_size, width_semantic, mapping=None, training_tensor='keras_learning_phase:0'):
    y_test = []
    pred = []
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = process_image_file(os.path.join(testfolder, line[1]), 0.08, input_size)
        x = x.astype('float32') / 255.0

        x1 = loadDataJSRTSingle(os.path.join(testfolder, line[1]), (width_semantic,width_semantic))
        y_test.append(mapping[line[2]])
        pred_values = sess.run(pred_tensor, feed_dict={input_tensor: np.expand_dims(x, axis=0), 
                                                       input_semantic_tensor: np.expand_dims(x1, axis=0),
                                                       training_tensor: 0})
        pred.append(np.array(pred_values).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens', ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    print('PPV', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))