from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file


def eval(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size,mapping=None):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)

    y_test = []
    pred = []
    for i in range(testfile.shape[0]):
        line = testfile[i]
        if(line[2] == "None"):
            continue
        x = process_image_file(os.path.join(testfolder, line[1]), 0.08, input_size)
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[2]])
        pred.append(np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})).argmax(axis=1))
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
        print('class_acc: {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                                 class_acc[0],
                                                                 list(mapping.keys())[list(mapping.values()).index(1)],
                                                                 class_acc[1],
                                                                 list(mapping.keys())[list(mapping.values()).index(2)],
                                                                 class_acc[2]))

    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    try:
        print('ppvs: {}: {}, {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)], ppvs[0],
                                                  list(mapping.keys())[list(mapping.values()).index(1)], ppvs[1],
                                                  list(mapping.keys())[list(mapping.values()).index(2)], ppvs[2],
                                                  list(mapping.keys())[list(mapping.values()).index(3)], ppvs[3]))
    except:
        print('ppvs: {}: {}, {}: {}, {}: {}'.format(list(mapping.keys())[list(mapping.values()).index(0)],
                                                            ppvs[0],
                                                            list(mapping.keys())[list(mapping.values()).index(1)],
                                                            ppvs[1],
                                                            list(mapping.keys())[list(mapping.values()).index(2)],
                                                            ppvs[2]))



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
