import numpy as np
import glob
import os
import cv2
from skimage import transform, io, img_as_float, exposure

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, 1).
It may be more convenient to store preprocessed data for faster loading.
Dataframe should contain paths to images and masks as two columns (relative to `path`).
"""

def loadDataJSRT(img_dir, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X, y,z = [], [],[]
    for imf in glob.glob(os.path.join(img_dir, '*.png')):
        img = io.imread(imf,as_gray=True)
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)
        X.append(img)
        y.append(os.path.basename(imf))
        z.append(io.imread(imf,as_gray=True))
    X = np.array(X)
    X -= X.mean()
    X /= X.std()
    return X, y,z

def loadDataJSRTSingle(img_dir, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X= []
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, im_shape)
    img = np.expand_dims(img, -1)
    X.append(img)

    X = np.array(X)
    X=X.astype('float64')
    X -= X.mean()
    X /= X.std()
    return X[0]

