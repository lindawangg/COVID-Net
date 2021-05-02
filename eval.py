from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
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


def eval(sess, model_semantic, testfile, testfolder, input_tensor, input_semantic_tensor,
         pred_tensor, input_size, width_semantic, mapping=None, training_tensor='keras_learning_phase:0'):
    y_test = []
    pred = []
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = process_image_file(os.path.join(testfolder, line[1]), 0.08, input_size)
        x = x.astype('float32') / 255.0

        x1 = loadDataJSRTSingle(os.path.join(testfolder, line[1]), (width_semantic, width_semantic))
        # x1= np.zeros((256,256,1)).astype('float32')
        y_test.append(mapping[line[2]])
        pred_values = sess.run(pred_tensor, feed_dict={input_tensor: np.expand_dims(x, axis=0), 
                                                       input_semantic_tensor: np.expand_dims(x1, axis=0),
                                                       training_tensor: False,
                                                       K.learning_phase(): 1})
        pred.append(np.array(pred_values).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    # Create confusion matrix
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)

    # Compute accuracy, sensitivity, and PPV
    diag = np.diag(matrix)
    acc = diag.sum()/max(matrix.sum(), 1)
    sens = diag/np.maximum(matrix.sum(axis=1), 1)
    ppv = diag/np.maximum(matrix.sum(axis=0), 1)
    print('Accuracy -', '{:.3f}'.format(acc))
    print('Sens -', ', '.join('{}: {:.3f}'.format(cls.capitalize(), sens[i]) for cls, i in mapping.items()))
    print('PPV -', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppv[i]) for cls, i in mapping.items()))

    # Store results in dict
    metrics = {'sens_' + cls: sens[i] for cls, i in mapping.items()}
    metrics.update({'ppv_' + cls: ppv[i] for cls, i in mapping.items()})
    metrics['accuracy'] = acc

    return metrics
