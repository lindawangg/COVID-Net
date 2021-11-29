# COVIDx Dataset
**Update 11/26/2021:Released a new training dataset with over 30,000 CXR images from a multinational cohort of over 16,400 patients. The dataset contains 16,490 positive COVID-19 images from over 2,800 patients. The COVIDx V9A dataset is for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia, and COVIDx V9B dataset is for COVID-19 positive/negative detection.**\
**Update 04/21/2021:Released COVIDxSev, a new airspace severity grading dataset for COVID-19 positive patients for COVIDNet CXR-S model.**\
**Update 03/19/2021:Released new datasets with both over 16,000 CXR images from a multinational cohort of over 15,100 patients from at least 51 countries. The dataset contains over 2,300 positive COVID-19 images from over 1,500 patients. The COVIDx V8A dataset is for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia, and COVIDx V8B dataset is for COVID-19 positive/negative detection.**\
**Update 01/28/2021:Released new datasets with over 15600 CXR images and over 1700 positive COVID-19 images. The COVIDx V7A dataset is for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia, and COVIDx V7B dataset is for COVID-19 positive/negative detection.**\
**Update 01/05/2021: Released new dataset for binary classification (COVID-19 positive or COVID-19 negative). Train dataset contains 517 positive and 13794 negative samples. Test dataset contains 100 positive and 100 negative samples.**\
**Update 10/30/2020: Released new dataset containing 517 COVID-19 train samples. Test dataset remains the same for consistency.**\
**Update 06/26/2020: Released new dataset with over 14000 CXR images containing 473 COVID-19 train samples. Test dataset remains the same for consistency.**\
**Update 05/13/2020: Released new dataset with 258 COVID-19 train and 100 COVID-19 test samples. There are constantly new xray images being added to covid-chestxray-dataset, Figure1, Actualmed and COVID-19 radiography database so we included train_COVIDx3.txt and test_COVIDx3.txt, which are the xray images we used for training and testing of the CovidNet-CXR3 models.**

The current COVIDx dataset can be downloaded from the following open source site:
* https://www.kaggle.com/andyczhao/covidx-cxr2?select=competition_test

Or can be manually constructed through our dataset scripts using the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)
* https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281
* https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912
* https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/

<!--We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Actualmed, M.E.H. Chowdhury et al., Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.-->

## Steps to download the dataset directly
The latest COVIDx9 training and testing dataset can be downloaded directly from Kaggle using the following steps:
1. Download the complete train and test datasets for Covidx9 from the [COVIDx CXR-2 Kaggle Dataset](https://www.kaggle.com/andyczhao/covidx-cxr2?select=competition_test)

The version 5 train and test text files are compatible with the latest [train\_COVIDx9B.txt](../labels/train_COVIDx9B.txt) and [test\_COVIDx9B.txt](../labels/test_COVIDx9B.txt) label files for COVID-19 positive/negative detection, and [train\_COVIDx9A.txt](../labels/train_COVIDx9A.txt) and [test\_COVIDx9A.txt](../labels/test_COVIDx9A.txt) label files for for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.

 * [train\_COVIDx9A.txt](../labels/train_COVIDx9A.txt): This file contains the training labels for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [test\_COVIDx9A.txt](../labels/test_COVIDx9A.txt): This file contains the testing labels for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [train\_COVIDx9B.txt](../labels/train_COVIDx9B.txt): This file contains the training labels for COVID-19 positive/negative detection.
 * [test\_COVIDx9B.txt](../labels/test_COVIDx9B.txt): This file contains the testing labels for COVID-19 positive/negative detection.

## Steps to generate the dataset
The older COVIDx8 training and testing dataset can be reconstructed using the following steps:
1. Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git`
 * go to this [link](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/version/3) to download the COVID-19 Radiography database. Only the COVID-19 image folder and metadata file is required. The overlaps between covid-chestxray-dataset are handled in the dataset curation scripts. **Note:** for COVIDx versions 8 & 7 please use Version 3 of the dataset, and for versions COVIDx6 and below please use Version 1.
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
 * go to this [link] (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281) to download the RICORD COVID-19 dataset, clinical data csv, and annotations
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [create\_ricord\_dataset\\create\_ricord\_dataset.ipynb](../create_ricord_dataset/create_ricord_dataset.ipynb) to pre-process the RICORD dataset before handling.
3. Use [create\_COVIDx\_binary.ipynb](../create_COVIDx_binary.ipynb) to combine the three datasets to create COVIDx for binary classification. Make sure to remember to change the file paths. Use [create\_COVIDx.ipynb](../create_COVIDx.ipynb) for datasets compatible with COVIDx5 and earlier models (not binary classification).
4. We provide the train and test txt files with patientId, image path and label. Note that the label is 'positive' or 'negative' for COVIDx8B and later or 'normal', 'pneumonia', and 'COVID-19' for COVIDx8A and COVIDx5 and earlier datasets. The description for each file is explained below:
 * [train\_COVIDx8A.txt](../labels/train_COVIDx8A.txt): This file contains the samples used for training COVIDNet-CXR for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [test\_COVIDx8A.txt](../labels/test_COVIDx8A.txt): This file contains the samples used for testing COVIDNet-CXR for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [train\_COVIDx8B.txt](../labels/train_COVIDx8B.txt): This file contains the samples used for training COVIDNet-CXR for COVID-19 positive/negative detection.
 * [test\_COVIDx8B.txt](../labels/test_COVIDx8B.txt): This file contains the samples used for testing COVIDNet-CXR for COVID-19 positive/negative detection.

## Latest COVIDx data distribution
COVIDx V9B
Chest radiography images distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:-----------------:|:-----------------:|:-----:|
| train |       13992       |        16490      | 30482 |
|  test |        200        |        200        |  400  |

Patients distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:-----------------:|:-----------------:|:-----:|
| train |       13850       |        2808       | 16648 |
|  test |        200        |         178       |  378  |


COVIDx V9A 
Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  8085  |    5555   |   16490  | 30130 |
|  test |   100  |     100   |   200    |   400 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  8085  |    5531   |   2808   |  16424 |
|  test |   100  |     100   |    178   |    378 |
