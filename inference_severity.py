import numpy as np
import tensorflow as tf
import os, argparse

from data import process_image_file
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
            self.output_tr = self.graph.get_tensor_by_name('MLP/dense_1/MatMul:0')

    def infer(self, image):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Lung Severity Scoring')
    parser.add_argument('--weightspath_geo', default='models/COVIDNet-SEV-GEO', type=str, help='Path to output folder')
    parser.add_argument('--weightspath_opc', default='models/COVIDNet-SEV-OPC', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to perfom scoring on')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

    args = parser.parse_args()

    x = process_image_file(args.imagepath, args.top_percent, args.input_size)
    x = x.astype('float32') / 255.0

    # check if models exists
    infer_geo = os.path.exists(os.path.join(args.weightspath_geo, args.metaname))
    infer_opc = os.path.exists(os.path.join(args.weightspath_opc, args.metaname))

    if infer_geo:
        model_geo = MetaModel(os.path.join(args.weightspath_geo, args.metaname),
                              os.path.join(args.weightspath_geo, args.ckptname))
        output_geo = model_geo.infer(x)

        print('Geographic severity: {:.3f}'.format(output_geo[0]))
        print('Geographic extent score for right + left lung (0 - 8): {:.3f}'.format(output_geo[0]*8))
        print('For each lung: 0 = no involvement; 1 = <25%; 2 = 25-50%; 3 = 50-75%; 4 = >75% involvement.')

    if infer_opc:
        model_opc = MetaModel(os.path.join(args.weightspath_opc, args.metaname),
                              os.path.join(args.weightspath_opc, args.ckptname))
        output_opc = model_opc.infer(x)

        print('Opacity severity: {:.3f}'.format(output_opc[0]))
        print('Opacity extent score for right + left lung (0 - 8): {:.3f}'.format(output_opc[0]*8))
        print('For each lung, the score is from 0 to 4, with 0 = no opacity and 4 = white-out.')

    print('**DISCLAIMER**')
    print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
