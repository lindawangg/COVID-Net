from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os, argparse

from data import (
    process_image_file, 
    process_image_file_medusa,
)

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_metrics(y_test, pred, mapping):
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)

    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

    print('Sens', ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
    print('PPV', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))


def eval(
    sess, 
    graph, 
    testfile, 
    testfolder, 
    input_tensor, 
    output_tensor, 
    input_size, 
    mapping, 
    is_medusa_backbone=False,
    medusa_input_tensor="input_1:0",
    medusa_input_size=256, 
):
    y_test = []
    pred = []

    for i in range(len(testfile)):
        line = testfile[i].split()
        image_file = os.path.join(testfolder, line[1])

        y_test.append(mapping[line[2]])

        if is_medusa_backbone:
            x = process_image_file(image_file, input_size, top_percent=0, crop=False)
            x = x.astype('float32') / 255.0
            medusa_x = process_image_file_medusa(image_file, medusa_input_size)
            feed_dict = {
                medusa_input_tensor: np.expand_dims(medusa_x, axis=0),
                input_tensor: np.expand_dims(x, axis=0),
            }
        else:
            x = process_image_file(image_file, input_size, top_percent=0.08)
            x = x.astype('float32') / 255.0
            feed_dict = {input_tensor: np.expand_dims(x, axis=0)}
        
        pred.append(np.array(sess.run(output_tensor, feed_dict=feed_dict)).argmax(axis=1))
    
    y_test = np.array(y_test)
    pred = np.array(pred)

    print_metrics(y_test, pred, mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
    parser.add_argument('--weightspath', default='models/COVIDNet-CXR-3', type=str, 
                    help='Path to model files, defaults to \'models/COVIDNet-CXR-3\'')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes, defaults to 2')
    parser.add_argument('--testfile', default='labels/test_COVIDx9B.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_2:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--in_tensorname_medusa', default='input_1:0', type=str, 
                    help='Name of input tensor to MEDUSA graph for COVIDNet-CXR-3')
    parser.add_argument('--out_tensorname', default='softmax/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--input_size_medusa', default=256, type=int, 
                    help='Size of input to MEDUSA graph (ex: if 256x256, --input_size 256)')
    parser.add_argument('--is_severity_model', action='store_true', help='Add flag if training COVIDNet CXR-S model')
    parser.add_argument('--is_medusa_backbone', action='store_true', 
                    help='Add flag if training COVIDNet CXR-3 model, do not include for other versions')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    graph = tf.get_default_graph()

    file = open(args.testfile, 'r')
    testfile = file.readlines()

    if args.is_severity_model:
        # For COVIDNet CXR-S training with COVIDxSev level 1 and level 2 air space seveirty grading
        mapping = {
            'level2': 0,
            'level1': 1
        }
    elif args.n_classes == 2:
        # For COVID-19 positive/negative detection
        mapping = {
            'negative': 0,
            'positive': 1,
        }
    elif args.n_classes == 3:
        # For detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
        mapping = {
            'normal': 0,
            'pneumonia': 1,
            'COVID-19': 2
        }
    else:
        raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection
            or 3 class detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia''')

    eval(
        sess, 
        graph, 
        testfile, 
        args.testfolder,
        args.in_tensorname, 
        args.out_tensorname,
        args.input_size, 
        mapping,
        is_medusa_backbone=args.is_medusa_backbone,
        medusa_input_tensor=args.in_tensorname_medusa,
        medusa_input_size=args.input_size_medusa,
    )
