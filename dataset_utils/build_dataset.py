from dataset_utils.especial import sirm_image_handling
from dataset_utils.utils import download_file_from_google_drive, remove_sub_dir
import zipfile
import pandas as pd
import pydicom as dicom
from shutil import copyfile
import numpy as np
import cv2
import os


class BuildDataset:

    def __init__(self, root_directory, mapping, urls_csv, urls_dataset, dataset_meta, train_split=0.1,
                 test_specials=None, label_list=[], complete_set=False):
        self.train = []
        self.test = []
        _, self.test_count = self.create_labels(label_list)
        _, self.train_count = self.create_labels(label_list)
        self.root_directory = root_directory
        self.mapping = mapping
        self.test_specials = test_specials
        self.labels, self.count_labels = self.create_labels(label_list)
        self.dataset_files = {}
        self.csv_files = {}
        self.complete_set = complete_set
        self.dataset_meta = dataset_meta
        self.dataset_names = list(dataset_meta.keys())
        self.create_datasets(urls_dataset)
        self.create_csv(urls_csv)
        self.process_csv_file()
        self.train_test_split()
        self.create_train_test_files()

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

    def create_labels(self, label_list):
        labels = {}
        count_labels = {}
        for label in label_list:
            labels[label] = []
            count_labels[label] = 0
        return labels, count_labels

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
                        self.count_labels[self.mapping[f]] += 1
                        image_name = self.check_image_exist(self.dataset_files[element],
                                                            self.dataset_meta[element]['image_name'], row)
                        if image_name is not None:
                            entry = [row[self.dataset_meta[element]['patientid']], image_name, self.mapping[f], element]
                            self.labels[self.mapping[f]].append(entry)

    # This is for handling special cases
    def check_image_exist(self, path, image_name, row):
        if ("png" in row[image_name]):
            return row[image_name]
        elif ("jpg" in row[image_name]):
            return row[image_name]
        else:
            if os.path.exists(os.path.join(path, row[image_name] + '.jpg')):
                return row[image_name] + '.jpg'
            elif os.path.exists(os.path.join(path, row[image_name] + '.png')):
                return row[image_name] + '.png'
            elif (self.complete_set):
                raise Exception(
                    'Missing Image. Please check if all csv images exists or change the complete flag')
            else:
                return None

    def download_url(self, url, name, extension):
        name = name + "." + extension
        file_name = download_file_from_google_drive(url, self.root_directory, name)
        return file_name

    def train_test_split(self):
        patient_imgpath = {}
        for key in self.labels.keys():
            arr = np.array(self.labels[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr[:,0]))
            # num_test = max(1, round(split*num_diff_patients))
            # select num_test number of random patients
            # random.sample(list(arr[:,0]), num_test)
            if(key in self.test_specials):
                test_patients=self.test_specials[key]
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
                        sirm_image_handling(patient,self.root_directory,self.dataset_files,test_flag=True)
                    else:
                        os.makedirs(os.path.dirname(os.path.join(self.root_directory, 'test', patient[1])), exist_ok=True)
                        copyfile(os.path.join(self.dataset_files[patient[3]], patient[1]),
                                 os.path.join(self.root_directory, 'test', patient[1]))
                    self.test.append(patient)
                    self.test_count[patient[2]] += 1
                else:
                    if patient[3] == 'sirm':
                        sirm_image_handling(patient,self.root_directory,self.dataset_files,test_flag=False)
                    else:
                        os.makedirs(os.path.dirname(os.path.join(self.root_directory, 'train', patient[1])), exist_ok=True)
                        copyfile(os.path.join(self.dataset_files[patient[3]], patient[1]),
                                 os.path.join(self.root_directory, 'train', patient[1]))
                    self.train.append(patient)
                    self.train_count[patient[2]] += 1

        print('test count: ', self.test_count)
        print('train count: ', self.train_count)


    def create_train_test_files(self):
        # export to train and test csv
        # format as patientid, filename, label, separated by a space
        train_file = open(os.path.join(self.root_directory,"train_split.txt"), 'w')
        for sample in self.train:
            if len(sample) == 4:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
            else:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            train_file.write(info)

        train_file.close()

        test_file = open(os.path.join(self.root_directory,"test_split.txt"), 'w')
        for sample in self.test:
            if len(sample) == 4:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
            else:
                info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            test_file.write(info)

        test_file.close()
