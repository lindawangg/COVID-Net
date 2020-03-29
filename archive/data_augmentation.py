import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(0)

def rotate_image(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def horizontal_flip(image):
    return cv2.flip(image, 1)

def shift_image(image, lr_pixels, tb_pixels):
    num_rows, num_cols = image.shape[:2]
    translation_matrix = np.float32([ [1,0,lr_pixels], [0,1,tb_pixels] ])
    return cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

INPUT_SIZE = (224, 224)
mapping = {'normal': 0, 'bacteria': 1, 'viral': 2, 'COVID-19': 3}
train_filepath = 'train_split.txt'
test_filepath = 'test_split.txt'
num_samples = 3000

# load in the train and test files
file = open(train_filepath, 'r')
trainfiles = file.readlines()
file = open(test_filepath, 'r')
testfiles = file.readlines()

# augment all the train class to 3000 examples each
# get number of each class
classes = {'normal': [], 'bacteria': [], 'viral': [], 'COVID-19': []}
img_aug = {'normal': [], 'bacteria': [], 'viral': [], 'COVID-19': []}
classes_test = {'normal': [], 'bacteria': [], 'viral': [], 'COVID-19': []}

for i in range(len(trainfiles)):
    train_i = trainfiles[i].split()
    classes[train_i[2]].append(train_i[1])
for i in range(len(testfiles)):
    test_i = testfiles[i].split()
    classes_test[test_i[2]].append(test_i[1])

for key in classes.keys():
    print('{}: {}'.format(key, len(classes[key])))

num_to_augment = {'normal': min(num_samples - (len(classes['normal']) + len(img_aug['normal'])), len(classes['normal'])),
                  'bacteria': min(num_samples - (len(classes['bacteria']) + len(img_aug['normal'])), len(classes['bacteria'])),
                  'viral': min(num_samples - (len(classes['viral']) + len(img_aug['normal'])), len(classes['viral'])),
                  'COVID-19': min(num_samples - (len(classes['COVID-19']) + len(img_aug['normal'])), len(classes['COVID-19']))}
print('num_to_augment 1:', num_to_augment)

to_augment = 0
for key in num_to_augment.keys():
    to_augment += num_to_augment[key]
print(to_augment)

while to_augment:
    for key in classes.keys():
        aug_class = classes[key]
        # sample which images to augment
        sample_indexes = np.random.choice(len(aug_class), num_to_augment[key], replace=False)
        for i in sample_indexes:
            # randomly select the degree of each augmentation
            rot = np.random.uniform(-5, 5)
            do_flip = np.random.randint(0, 2)
            shift_vert = np.random.randint(0, 2)
            shift = np.random.uniform(-10, 10)
            # read in image and apply augmentation
            img = cv2.imread(os.path.join('data', 'train', aug_class[i]))
            #img = rotate_image(img, rot)
            #if shift_vert:
            #    img = shift_image(img, 0, shift)
            #else:
            #    img = shift_image(img, shift, 0)
            if do_flip:
                img = horizontal_flip(img)
            # append filename and class to img_aug, save as png
            imgname = '{}.png'.format(aug_class[i].split('.')[0] + '_aug_r' + str(round(rot)) + '_' + str(do_flip) + '_s' + str(shift_vert) + str(round(shift)))
            img_aug[key].append(imgname)
            cv2.imwrite(os.path.join('data', 'train', imgname), img)
    # update num_to_augment numbers
    num_to_augment = {
        'normal': min(num_samples - (len(classes['normal']) + len(img_aug['normal'])), len(classes['normal'])),
        'bacteria': min(num_samples - (len(classes['bacteria']) + len(img_aug['bacteria'])), len(classes['bacteria'])),
        'viral': min(num_samples - (len(classes['viral']) + len(img_aug['viral'])), len(classes['viral'])),
        'COVID-19': min(num_samples - (len(classes['COVID-19']) + len(img_aug['COVID-19'])), len(classes['COVID-19']))}
    to_augment = 0
    for key in num_to_augment.keys():
        to_augment += num_to_augment[key]
    print(num_to_augment)

mapping = {'normal': 0, 'bacteria': 1, 'viral': 2, 'COVID-19': 3}

train_file = open("train_augment.txt","a")
for key in classes.keys():
    for imgname in classes[key]:
        info = imgname + ' ' + str(mapping[key]) + '\n'
        train_file.write(info)
    for imgname in img_aug[key]:
        info = imgname + ' ' + str(mapping[key]) + '\n'
        train_file.write(info)

train_file.close()

test_file = open("test.txt", "a")
for key in classes_test.keys():
    for imgname in classes_test[key]:
        info = imgname + ' ' + str(mapping[key]) + '\n'
        test_file.write(info)
test_file.close()
