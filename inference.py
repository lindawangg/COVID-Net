import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import (
    process_image_file,
    process_image_file_medusa,
)

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR-3', type=str, 
                    help='Path to model files, defaults to \'models/COVIDNet-CXR-3\'')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes, defaults to 2')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_2:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--in_tensorname_medusa', default='input_1:0', type=str, 
                    help='Name of input tensor to MEDUSA graph for COVIDNet-CXR-3')
parser.add_argument('--out_tensorname', default='softmax/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--input_size_medusa', default=256, type=int, 
                    help='Size of input to MEDUSA graph (ex: if 256x256, --input_size 256)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--is_severity_model', action='store_true', help='Add flag if training COVIDNet CXR-S model')
parser.add_argument('--is_medusa_backbone', action='store_true', 
                    help='Add flag if training COVIDNet CXR-3 model, do not include for other versions')

args = parser.parse_args()

if args.is_severity_model:
    # For COVIDNet CXR-S training with COVIDxSev level 1 and level 2 air space seveirty grading
    mapping = {'level2': 0, 'level1': 1}
    inv_mapping = {0: 'level2', 1: 'level1'}
elif args.n_classes == 2:
    # For COVID-19 positive/negative detection
    mapping = {'negative': 0, 'positive': 1}
    inv_mapping = {0: 'negative', 1: 'positive'}
elif args.n_classes == 3:
    # For detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
else:
    raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection
        or 3 class detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia''')
mapping_keys = list(mapping.keys())

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(args.in_tensorname)
pred_tensor = graph.get_tensor_by_name(args.out_tensorname)

if args.is_medusa_backbone:
    x = process_image_file(args.imagepath, args.input_size, top_percent=0, crop=False)
    x = x.astype('float32') / 255.0
    medusa_image_tensor = graph.get_tensor_by_name(args.in_tensorname_medusa)
    medusa_x = process_image_file_medusa(args.imagepath, args.input_size_medusa)
    feed_dict = {
                medusa_image_tensor: np.expand_dims(medusa_x, axis=0),
                image_tensor: np.expand_dims(x, axis=0),
            } 
else:
    x = process_image_file(args.imagepath, args.input_size, top_percent=args.top_percent)
    x = x.astype('float32') / 255.0
    feed_dict = {image_tensor: np.expand_dims(x, axis=0)}

pred = sess.run(pred_tensor, feed_dict=feed_dict)

print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('Confidence')
print(' '.join('{}: {:.3f}'.format(cls.capitalize(), pred[0][i]) for cls, i in mapping.items()))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
