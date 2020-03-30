from sklearn.metrics import confusion_matrix
import numpy as np
import keras
import os, argparse
import cv2

from model import build_COVIDNet

parser = argparse.ArgumentParser(description='COVID-Net Evaluation for Keras')
parser.add_argument('--checkpoint', default='', type=str, help='Start training from existing weights')
parser.add_argument('--testfile', default='test_COVIDx.txt', type=str, help='Name of testfile')
parser.add_argument('--testfolder', default='test', type=str, help='Folder where test data is located')

args = parser.parse_args()
mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

file = open(args.testfile, 'r')
testfiles = file.readlines()

model = build_COVIDNet(checkpoint=args.checkpoint)

y_test = []
pred = []
for i in range(len(testfiles)):
    line = testfiles[i].split()
    x = cv2.imread(os.path.join('data', 'test', line[1]))
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0
    y_test.append(mapping[line[2]])
    pred.append(np.array(model.predict(np.expand_dims(x, axis=0))).argmax(axis=1))
y_test = np.array(y_test)
pred = np.array(pred)

matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
print(matrix)
class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                           class_acc[1],
                                                                           class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                         ppvs[1],
                                                                         ppvs[2]))
