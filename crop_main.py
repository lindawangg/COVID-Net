import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from load_data import loadDataJSRT
from model.build_model import build_UNet2D_4L



if __name__ == '__main__':
    img_dir = '/Users/hosseinaboutalebi/Desktop/negative_hhs'
    out_dir = '/Users/hosseinaboutalebi/Desktop/croped_check'
    height=256
    width=256
    os.makedirs(out_dir, exist_ok=True)
    model=build_UNet2D_4L((height,width,1))
    model.load_weights("./saved_model/trained_model.hdf5")

    saving = True

    lines = []
    counter =0
    X_val,image_names,z = loadDataJSRT(img_dir, (height,width))
    for i in range(len(X_val)):
        fname = image_names[i]
        img=np.expand_dims(X_val[i],axis=0)
        post_process_img= model.predict(img)
        out_path = os.path.join(out_dir, fname)
        print("processeing image number {}".format(counter))
        try:
            post_process_img=np.squeeze(post_process_img,axis=0)
            post_process_img=np.squeeze(post_process_img, axis=2)
            cv2.imwrite(out_path, post_process_img*256)
            plt.subplot(121)
            plt.imshow(post_process_img*256, cmap='gray')
            plt.subplot(122)
            plt.imshow(z[i], cmap='gray')
            plt.savefig(out_path)
            plt.show()
        except:
            print('did not work on this one for save: ', fname)

    if saving:
        with open(os.path.join(out_dir, 'bboxes.txt'), 'w') as f:
            f.writelines(lines)