import numpy as np
import tensorflow as tf
import os, argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

from data import process_image_file, _process_csv_file
from collections import defaultdict

def score_prediction(softmax, step_size):
    vals = np.arange(3) * step_size + (step_size / 2.)
    vals = np.expand_dims(vals, axis=0)
    return np.sum(softmax * vals, axis=-1)

class MetaModel:
    def __init__(self, meta_file, ckpt_file):
        self.meta_file = meta_file
        self.ckpt_file = ckpt_file

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.meta_file)
            self.input_tr = self.graph.get_tensor_by_name('input_1:0')
            self.phase_tr = self.graph.get_tensor_by_name('keras_learning_phase:0')
            #self.output_tr = self.graph.get_tensor_by_name('MLP/dense_1/MatMul:0')
            self.output_tr = self.graph.get_tensor_by_name('MatMul:0')

    def infer_single_image(self, image):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, self.ckpt_file)

            outputs = defaultdict(list)
            outs = sess.run(self.output_tr,
                            feed_dict={
                                self.input_tr: np.expand_dims(image, axis=0),
                                self.phase_tr: False
                            })
            outputs['logits'].append(outs)

            for k in outputs.keys():
                outputs[k] = np.concatenate(outputs[k], axis=0)

            outputs['softmax'] = np.exp(outputs['logits']) / np.sum(
                np.exp(outputs['logits']), axis=-1, keepdims=True)
            outputs['score'] = score_prediction(outputs['softmax'], 1 / 3.)

        return outputs['score']

    def infer_multiple_images(self, image_path_dict):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, self.ckpt_file)

            output_dict = {}
            for key, image_path in image_path_dict.items():
                x = process_image_file(image_path, args.top_percent, args.input_size)
                image = x.astype('float32') / 255.0

                outputs = defaultdict(list)
                outs = sess.run(self.output_tr,
                                feed_dict={
                                    self.input_tr: np.expand_dims(image, axis=0),
                                    self.phase_tr: False
                                })
                #outputs['logits'].append(outs)

                #for k in outputs.keys():
                #    outputs[k] = np.concatenate(outputs[k], axis=0)

                #outputs['softmax'] = np.exp(outputs['logits']) / np.sum(
                #    np.exp(outputs['logits']), axis=-1, keepdims=True)
                #outputs['score'] = score_prediction(outputs['softmax'], 1 / 3.)
                #output_dict[key] = outputs['score']
                output_dict[key] = outs
        return output_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Lung Severity Scoring')
    parser.add_argument('--weightspath_geo', default='output/COVIDNet-MNTF-Sev-geo-lr2e-05', type=str, help='Path to output folder')
    parser.add_argument('--weightspath_opc', default='models/COVIDNet-SEV-OPC', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--image_dir', default='../montefiore_severity/CXR/', type=str, help='Full path to set of images to perfom scoring on')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
    parser.add_argument('--labels_file', default='test_mntf_sev.txt',type=str, help='File with labels for evaluation - if specified, only these images in imagedir will be tested, else all files in imagedir will be evaluated.')

    args = parser.parse_args()

    # check if models exists
    infer_geo = os.path.exists(os.path.join(args.weightspath_geo, args.metaname))
    infer_opc = os.path.exists(os.path.join(args.weightspath_opc, args.metaname))

    if args.labels_file is not None:
        # get paths for images specified in labels_file
        image_csv = _process_csv_file(args.labels_file)
        image_paths = []
        geo_labels = {}
        opc_labels = {}
        for l in image_csv:
            splits = l.split() # form is <img.jpg subject_id geo_mean opc_mean>
            image_paths.append(os.path.join(args.image_dir, splits[0]))
            geo_labels[splits[0]] = splits[2]
            opc_labels[splits[0]] = splits[3]
    else:
        # get paths for all images in image_dir
        image_paths = glob.glob(os.path.join(args.image_dir, '*'))

    # troubleshoot with subset of images
    # image_paths = image_paths[:100]
    image_path_dict = {}
    for image_path in image_paths:
        image_name = image_path.split('/')[-1]
        image_path_dict[image_name] = image_path

    if infer_geo:
        model_geo = MetaModel(os.path.join(args.weightspath_geo, args.metaname),
                              os.path.join(args.weightspath_geo, args.ckptname))
       
        geo_scores = model_geo.infer_multiple_images(image_path_dict)

        # collect scores and labels into np arrays
        g_s_pred = []
        g_s_labels = []
        for k, g_s in geo_scores.items():
            g_s_pred.append(g_s[0])
            g_s_labels.append(geo_labels[k])
        g_s_labels = np.array(g_s_labels, dtype=np.float16)
        g_s_pred = np.array(g_s_pred)

        # plot results vs labels with lines of best fit
        plt.scatter(g_s_labels, g_s_pred, label='Geo')
        m_geo, b_geo = np.polyfit(g_s_labels, g_s_pred, 1)
        plt.plot(g_s_labels, m_geo*g_s_labels + b_geo, label='Geo LOBF')

    if infer_opc:
        model_opc = MetaModel(os.path.join(args.weightspath_opc, args.metaname),
                              os.path.join(args.weightspath_opc, args.ckptname))
        opc_scores = model_geo.infer_multiple_images(image_path_dict)

        o_s_pred = []
        o_s_labels = []
        for k, o_s in opc_scores.items():
            o_s_pred.append(o_s[0])
            o_s_labels.append(opc_labels[k])
        o_s_labels = np.array(o_s_labels, dtype=np.float16)
        o_s_pred = np.array(o_s_pred)

        plt.scatter(o_s_labels, o_s_pred, label='Opc')
        m_opc, b_opc = np.polyfit(o_s_labels, o_s_pred, 1)
        plt.plot(o_s_labels, m_opc*o_s_labels + b_opc, label='Opacity LOBF')
        

    

    plt.xlabel('Radiologist Score')
    plt.ylabel('Predicted Score')
    plt.title('Geo and Opacity Score Pred vs Label')
    plt.legend()
    plt.show()
