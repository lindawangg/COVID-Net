from dataset_utils.utils import download_file_from_google_drive, remove_sub_dir
import zipfile
import pandas as pd
import pydicom as dicom
from shutil import copyfile
import numpy as np
import cv2
import os


class BuildDataset:

    def __init__(self, root_directory, mapping, urls_csv, urls_dataset, dataset_names, train_split=0.1,testspecials=None):
        self.root_directory = root_directory
        self.mapping = mapping
        self.testspecials=testspecials
        self.filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        self.count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        self.dataset_files = {}
        self.csv_files = {}
        self.dataset_names = dataset_names
        self.create_datasets(urls_dataset)
        self.create_csv(urls_csv)
        self.process_csv_file()

    def create_datasets(self, urls_dataset):
        for i in range(len(urls_dataset)):
            file_name = self.download_url(urls_dataset[i], self.dataset_names[i], extension="zip")
            dir_images = os.path.join(self.root_directory, self.dataset_names[i])
            if not os.path.exists(dir_images):
                os.makedirs(dir_images)
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(dir_images)
            os.remove(file_name)
            remove_sub_dir(dir_images)
            self.dataset_files[self.dataset_names[i]] = dir_images

    def create_csv(self, urls_csv):
        for i in range(len(urls_csv)):
            file_name = self.download_url(urls_csv[i], self.dataset_names[i], extension="csv")
            self.csv_files[self.dataset_names[i]] = file_name

    def process_csv_file(self):
        for element in self.csv_files:
            csv_df = pd.read_csv(self.csv_files[element], encoding='ISO-8859-1', nrows=None)
            for index, row in csv_df.iterrows():
                if not str(row['finding']) == 'nan':
                    f = row['finding'].split(',')[0]  # take the first finding
                    if f in self.mapping:  #
                        self.count[self.mapping[f]] += 1
                        if os.path.exists(os.path.join(self.dataset_files[element], row['patientid'] + '.jpg')):
                            entry = [row['patientid'], row['patientid'] + '.jpg', self.mapping[f], 'fig1']
                        elif os.path.exists(os.path.join(self.dataset_files[element], row['patientid'] + '.png')):
                            entry = [row['patientid'], row['patientid'] + '.png', self.mapping[f], 'fig1']
                        self.filename_label[self.mapping[f]].append(entry)

    def download_url(self, url, name, extension):
        name = name + "." + extension
        file_name = download_file_from_google_drive(url, self.root_directory, name)
        return file_name

    def train_test_split(self):
        train = []
        test = []
        test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        patient_imgpath={}
        for key in self.filename_label.keys():
            arr = np.array(self.filename_label[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr[:,0]))
            # num_test = max(1, round(split*num_diff_patients))
            # select num_test number of random patients
            # random.sample(list(arr[:,0]), num_test)
            if key == 'pneumonia':
                test_patients = self.testspecials['pneumonia']
            elif key == 'COVID-19':
                test_patients = self.testspecials['COVID-19']
            else:
                test_patients = []
            print('Key: ', key)
            print('Test patients: ', test_patients)
            # go through all the patients
            for patient in arr:
                if patient[0] not in patient_imgpath:
                    patient_imgpath[patient[0]] = [patient[1]]
                else:
                    if patient[1] not in patient_imgpath[patient[0]]:
                        patient_imgpath[patient[0]].append(patient[1])
                    else:
                        continue  # skip since image has already been written
                if patient[0] in test_patients:
                    if patient[3] == 'sirm':
                        image = cv2.imread(os.path.join(self.dataset_files[patient[3]], patient[1]))
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        patient[1] = patient[1].replace(' ', '')
                        cv2.imwrite(os.path.join(self.root, 'test', patient[1]), gray)
                    else:
                        copyfile(os.path.join(self.dataset_files[patient[3]], patient[1]),
                                 os.path.join(self.root, 'test', patient[1]))
                    test.append(patient)
                    test_count[patient[2]] += 1
                else:
                    if patient[3] == 'sirm':
                        image = cv2.imread(os.path.join(self.dataset_files[patient[3]], patient[1]))
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        patient[1] = patient[1].replace(' ', '')
                        cv2.imwrite(os.path.join(self.root, 'train', patient[1]), gray)
                    else:
                        copyfile(os.path.join(self.dataset_files[patient[3]], patient[1]),
                                 os.path.join(self.root, 'train', patient[1]))
                    train.append(patient)
                    train_count[patient[2]] += 1

        print('test count: ', test_count)
        print('train count: ', train_count)

    def create_train_test_files(self,train,test):
        # export to train and test csv
        # format as patientid, filename, label, separated by a space
        train_file = open("train_split.txt", 'w')
        for sample in train:
            if len(sample) == 4:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
            else:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            train_file.write(info)

        train_file.close()

        test_file = open("test_split.txt", 'w')
        for sample in test:
            if len(sample) == 4:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
            else:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            test_file.write(info)

        test_file.close()


