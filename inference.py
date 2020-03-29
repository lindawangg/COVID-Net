import numpy as np
import tensorflow as tf
import os, argparse
import cv2

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--weightspath', default='output', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')

args = parser.parse_args()

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name("input_1:0")
pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

x = cv2.imread(args.imagepath)
x = cv2.resize(x, (224, 224))
x = x.astype('float32') / 255.0
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
