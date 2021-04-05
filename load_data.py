import numpy as np
import glob
import os
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


def loadDataMontgomery(df, path, im_shape):
    """Function for loading Montgomery dataset"""
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        gt = io.imread(path + item[1])
        l, r = np.where(img.sum(0) > 1)[0][[0, -1]]
        t, b = np.where(img.sum(1) > 1)[0][[0, -1]]
        img = img[t:b, l:r]
        mask = gt[t:b, l:r]
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()
    return X, y


def loadDataGeneral(df, path, im_shape):
    """Function for loading arbitrary data in standard formats"""
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        mask = io.imread(path + item[1])
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()
    return X, y