import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file

parser = argparse.ArgumentParser(description='COVID-Net-P Inference')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

args = parser.parse_args()

#Combine the COVID and non-COVID pneumonia predictions
mapping = {'normal': 0, 'pneumonia': 1}
inv_mapping = {0: 'normal', 1: 'pneumonia'}

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(args.in_tensorname)
pred_tensor = graph.get_tensor_by_name(args.out_tensorname)

x = process_image_file(args.imagepath, args.input_size, top_percent=args.top_percent)
x = x.astype('float32') / 255.0
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
# Combining pneumonia and covid predictions into single pneumonia prediction.
pred_pneumonia = np.array([pred[0][0], np.max([pred[0][1], pred[0][2]])])
pred_pneumonia = pred_pneumonia / np.sum(pred_pneumonia)

print('Prediction: {}'.format(inv_mapping[pred_pneumonia.argmax()]))
print('Confidence')
print('Normal: {:.3f}, Pneumonia: {:.3f}'.format(pred_pneumonia[0], pred_pneumonia[1]))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
