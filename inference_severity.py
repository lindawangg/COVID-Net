import numpy as np
import tensorflow as tf
import os, argparse

from data import process_image_file
from collections import defaultdict

def cat2val_with_step(preds, step_size):
    return (preds + 0.5) * step_size

def cat2val_with_weights(probs, step_size):
    vals = np.arange(3) * step_size + (step_size / 2.)
    vals = np.expand_dims(vals, axis=0)
    return np.sum(probs * vals, axis=-1)

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

            outputs['probs'] = np.exp(outputs['logits']) / np.sum(
                np.exp(outputs['logits']), axis=-1, keepdims=True)
            outputs['preds'] = np.argmax(outputs['probs'], axis=-1)
            outputs['dvals'] = cat2val_with_step(outputs['preds'], 1 / 3.)
            outputs['wvals'] = cat2val_with_weights(outputs['probs'], 1 / 3.)

        return outputs['wvals']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Lung Severity Scoring')
    parser.add_argument('--weightspath', default='models/COVIDNet-SEV-GEO', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to perfom scoring on')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

    args = parser.parse_args()

    x = process_image_file(args.imagepath, args.top_percent, args.input_size)
    x = x.astype('float32') / 255.0

    model = MetaModel(os.path.join(args.weightspath, args.metaname),
                      os.path.join(args.weightspath, args.ckptname))
    output = model.infer(x)

    if 'GEO' in args.weightspath:
        print('Geographic severity: {:.3f}'.format(output[0]))
        print('Geographic extent score for right + left lung (0 - 8): {:.3f}'.format(output[0]*8))
        print('For each lung: 0 = no involvement; 1 = <25%; 2 = 25-50%; 3 = 50-75%; 4 = >75% involvement.')
    elif 'OPC' in args.weightspath:
        print('Opacity severity: {:.3f}'.format(output[0]))
        print('Opacity extent score for right + left lung (0 - 6): {:.3f}'.format(output[0]*6))
        print('For each lung: 0 = no opacity; 1 = ground glass opacity; 2 =consolidation; 3 = white-out.')
    else:
        print('Severity (0 - 1): {:.3f}'.format(output[0]))

    print('**DISCLAIMER**')
    print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
