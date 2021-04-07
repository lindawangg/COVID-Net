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

def eval(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size,width_semantic,mapping=None):
    input_2 = tf.placeholder(tf.float32)
    pred_tensor = output_tensor
    print("hooray")
    y_test = []
    pred = []
    for i in range(testfile.shape[0]):
        line = testfile[i]
        if(line[2] == "None"):
            continue
        x = fix_input_image(os.path.join(testfolder, line[1]), input_size, width_semantic,0.08)
        y_test.append(mapping[line[2]])
        pred.append(np.array(sess.run(pred_tensor, feed_dict={input_2: np.expand_dims(x, axis=0)})).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    try:
        print('class_acc: {}: {}, {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],class_acc[0],
                                                              list(mapping.keys())[list(mapping.values()).index(1)],class_acc[1],
                                                              list(mapping.keys())[list(mapping.values()).index(2)],class_acc[2],
                                                              list(mapping.keys())[list(mapping.values()).index(3)],class_acc[3]))
    except:
        try:
            print('class_acc: {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                                 class_acc[0],
                                                                 list(mapping.keys())[list(mapping.values()).index(1)],
                                                                 class_acc[1],
                                                                 list(mapping.keys())[list(mapping.values()).index(2)],
                                                                 class_acc[2]))
        except:
            print('class_acc: {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                             class_acc[0],
                                                             list(mapping.keys())[list(mapping.values()).index(1)],
                                                             class_acc[1]))


    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    try:
        print('ppvs: {}: {}, {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)], ppvs[0],
                                                  list(mapping.keys())[list(mapping.values()).index(1)], ppvs[1],
                                                  list(mapping.keys())[list(mapping.values()).index(2)], ppvs[2],
                                                  list(mapping.keys())[list(mapping.values()).index(3)], ppvs[3]))
    except:
        try:
            print('ppvs: {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                            ppvs[0],
                                                            list(mapping.keys())[list(mapping.values()).index(1)],
                                                            ppvs[1],
                                                            list(mapping.keys())[list(mapping.values()).index(2)],
                                                            ppvs[2]))
        except:
            print('ppvs: {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                        ppvs[0],
                                                        list(mapping.keys())[list(mapping.values()).index(1)],
                                                        ppvs[1]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
    parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
    parser.add_argument('--testfile', default='test_COVIDx5.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    graph = tf.get_default_graph()

    file = open(args.testfile, 'r')
    testfile = file.readlines()

    eval(sess, graph, testfile, args.testfolder, args.in_tensorname, args.out_tensorname, args.input_size)
