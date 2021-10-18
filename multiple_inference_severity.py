import numpy as np
import tensorflow as tf
import os, argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from data import process_image_file, _process_csv_file
from collections import defaultdict

def score_prediction(softmax, step_size):
    vals = np.arange(3) * step_size + (step_size / 2.)
    vals = np.expand_dims(vals, axis=0)
    return np.sum(softmax * vals, axis=-1)

class MetaModel:
    def __init__(self, meta_file, ckpt_file, old_model=False):
        self.meta_file = meta_file
        self.ckpt_file = ckpt_file
        # old_model is original one from Maya
        self.old_model = old_model

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.meta_file)
            if self.old_model:
                self.input_tr = self.graph.get_tensor_by_name('input_1:0')
                self.phase_tr = self.graph.get_tensor_by_name('keras_learning_phase:0')
                self.output_tr = self.graph.get_tensor_by_name('MLP/dense_1/MatMul:0')
            else:
                self.input_tr = self.graph.get_tensor_by_name('input_1:0')
                # NOTE: change this to the output tensor for your model
                self.output_tr = self.graph.get_tensor_by_name('regr_head/Sigmoid:0')

    def infer_single_image(self, image):
        # NOTE: not yet updated to "new" model type
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

                if self.old_model:
                    # Maya's Keras model
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
                    output_dict[key] = outputs['score']
                else:
                    # COVID-Net style model
                    outs = sess.run(self.output_tr,
                                feed_dict={
                                    self.input_tr: np.expand_dims(image, axis=0),
                                })
                    output_dict[key] = outs[0]

        return output_dict


if __name__ == '__main__':
    # defaults are set to newer, montefiore based models
    parser = argparse.ArgumentParser(description='COVID-Net Lung Severity Scoring')
    parser.add_argument('--weightspath_geo', default='models/COVIDNet-MNTF-Sev-geo', type=str, help='Path to output folder')
    parser.add_argument('--weightspath_opc', default='models/COVIDNet-MNTF-Sev-opc', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--image_dir', default='/home/alex.maclean/montefiore_severity/CXR/', type=str, help='Full path to set of images to perfom scoring on')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
    parser.add_argument('--labels_file', type=str, help='File with labels for evaluation - if specified, only these images in imagedir will be tested, else all files in imagedir will be evaluated.')

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
        # get paths for all images in image_dir, no stats/graphs shown
        # (not implemented anymore)
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

    if infer_opc:
        model_opc = MetaModel(os.path.join(args.weightspath_opc, args.metaname),
                              os.path.join(args.weightspath_opc, args.ckptname))
        opc_scores = model_opc.infer_multiple_images(image_path_dict)

    # collect scores and labels into np arrays
    g_s_pred = []
    g_s_labels = []
    for k, g_s in geo_scores.items():
        g_s_pred.append(g_s[0])
        g_s_labels.append(geo_labels[k])
    g_s_labels = np.array(g_s_labels, dtype=np.float16) / 8.0
    g_s_pred = np.array(g_s_pred)
    g_mse = mean_squared_error(g_s_labels, g_s_pred)
    g_r2 = r2_score(g_s_labels, g_s_pred)

    o_s_pred = []
    o_s_labels = []
    for k, o_s in opc_scores.items():
        o_s_pred.append(o_s[0])
        o_s_labels.append(opc_labels[k])
    o_s_labels = np.array(o_s_labels, dtype=np.float16) / 8.0
    o_s_pred = np.array(o_s_pred)
    o_mse = mean_squared_error(o_s_labels, o_s_pred)
    o_r2 = r2_score(o_s_labels, o_s_pred)

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    ax = ax.flat

    # plot predictions and Lines of Best Fit
    ax[0].scatter(g_s_labels, g_s_pred, label='Geo')
    m_geo, b_geo = np.polyfit(g_s_labels, g_s_pred, 1)
    ax[0].plot(g_s_labels, m_geo*g_s_labels + b_geo, color='r', label='Geo LOBF')
    ax[0].legend()
    ax[0].set_xlabel('Radiologist Score')
    ax[0].set_ylabel('Predicted Score')
    ax[0].set_title(f'Geo Pred vs Label - MSE: {g_mse:.3f} - R^2: {g_r2:.3f}')
    ax[0].set_ylim(-0.05, 1.05)

    ax[1].scatter(o_s_labels, o_s_pred, label='Opc')
    m_opc, b_opc = np.polyfit(o_s_labels, o_s_pred, 1)
    ax[1].plot(o_s_labels, m_opc*o_s_labels + b_opc, color='r', label='Opacity LOBF')
    ax[1].legend()
    ax[1].set_xlabel('Radiologist Score')
    ax[1].set_ylabel('Predicted Score')
    ax[1].set_title(f'Opc Pred vs Label - MSE: {o_mse:.3f} - R^2: {o_r2:.3f}')
    ax[1].set_ylim(-0.05, 1.05)

    # plot residual plots
    g_s_residuals = g_s_pred - g_s_labels
    ax[2].scatter(g_s_labels, g_s_residuals, label='Geo Residuals')
    ax[2].legend()
    ax[2].set_xlabel('Radiologist Score')
    ax[2].set_ylabel('Residual Error')
    ax[2].set_title(f'Geo Residual Plot - Mean Error: {np.mean(g_s_residuals):.3f}')

    o_s_residuals = o_s_pred - o_s_labels
    ax[3].scatter(o_s_labels, o_s_residuals, label='Opc Residuals')
    ax[3].legend()
    ax[3].set_xlabel('Radiologist Score')
    ax[3].set_ylabel('Residual Error')
    ax[3].set_title(f'Opc Residual Plot - Mean Error: {np.mean(o_s_residuals):.3f}')

    # printing components of R^2 calculation
    print('GEO RSS: ', np.sum(np.square(g_s_residuals)))
    print('GEO TSS: ', np.sum(np.square(np.mean(g_s_labels) - g_s_labels)))
    print('OPC RSS: ', np.sum(np.square(o_s_residuals)))
    print('OPC TSS: ', np.sum(np.square(np.mean(o_s_labels) - o_s_labels)))
    print(f'GEO Labels std: {np.std(g_s_labels)} - Pred std: {np.std(g_s_pred)}')
    print(f'OPC Labels std: {np.std(o_s_labels)} - Pred std: {np.std(o_s_pred)}')


    plt.tight_layout()
    fig.savefig('mntf_pred_and_residual_plot.png')
    plt.show()
