import cv2
import os

def sirm_image_handling(patient,root,dataset_files,test_flag):
    if(test_flag):
        class_image="test"
    else:
        class_image = "train"
    image = cv2.imread(os.path.join(dataset_files[patient[3]], patient[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patient[1] = patient[1].replace(' ', '')
    os.makedirs(os.path.dirname(os.path.join(root, class_image)), exist_ok=True)
    cv2.imwrite(os.path.join(root, class_image, patient[1]), gray)
