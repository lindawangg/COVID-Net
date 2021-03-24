# COVIDx Dataset
**Update 03/19/2021:Released new datasets with over 16,000 CXR images from a multinational cohort of 15,100 patients from at least 51 countries. Both datasets contains over 2,300 positive COVID-19 images from 1,500 patients. The COVIDx V8A dataset is for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia, and COVIDx V8B dataset is for COVID-19 positive/negative detection.**\
**Update 01/28/2021:Released new datasets with over 15600 CXR images and over 1700 positive COVID-19 images. The COVIDx V7A dataset is for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia, and COVIDx V7B dataset is for COVID-19 positive/negative detection.**\
**Update 01/05/2021: Released new dataset for binary classification (COVID-19 positive or COVID-19 negative). Train dataset contains 517 positive and 13794 negative samples. Test dataset contains 100 positive and 100 negative samples.**\
**Update 10/30/2020: Released new dataset containing 517 COVID-19 train samples. Test dataset remains the same for consistency.**\
**Update 06/26/2020: Released new dataset with over 14000 CXR images containing 473 COVID-19 train samples. Test dataset remains the same for consistency.**\
**Update 05/13/2020: Released new dataset with 258 COVID-19 train and 100 COVID-19 test samples. There are constantly new xray images being added to covid-chestxray-dataset, Figure1, Actualmed and COVID-19 radiography database so we included train_COVIDx3.txt and test_COVIDx3.txt, which are the xray images we used for training and testing of the CovidNet-CXR3 models.**

The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)
* https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281

<!--We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Actualmed, M.E.H. Chowdhury et al., Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.-->

## Steps to generate the dataset

1. Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git`
 * go to this [link](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) to download the COVID-19 Radiography database. Only the COVID-19 image folder and metadata file is required. The overlaps between covid-chestxray-dataset are handled
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
 * go to this [link] (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281) to download the RICORD COVID-19 dataset and clinical data csv
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [create\_ricord\_dataset\\create\_ricord\_dataset.ipynb](../create_ricord_dataset/create_ricord_dataset.ipynb) to pre-process the RICORD dataset before handling.
3. Use [create\_COVIDx\_binary.ipynb](../create_COVIDx_binary.ipynb) to combine the three datasets to create COVIDx for binary classification. Make sure to remember to change the file paths. Use [create\_COVIDx.ipynb](../create_COVIDx.ipynb) for datasets compatible with COVIDx5 and earlier models (not binary classification).
4. We provide the train and test txt files with patientId, image path and label. Note that the label is 'positive' or 'negative' for COVIDx8B and later or 'normal', 'pneumonia', and 'COVID-19' for COVIDx8A and COVIDx5 and earlier datasets. The description for each file is explained below:
 * [train\_COVIDx8A.txt](../labels/train_COVIDx8A.txt): This file contains the samples used for training COVIDNet-CXR for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [test\_COVIDx8A.txt](../labels/test_COVIDx8A.txt): This file contains the samples used for testing COVIDNet-CXR for detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.
 * [train\_COVIDx8B.txt](../labels/train_COVIDx8B.txt): This file contains the samples used for training COVIDNet-CXR for COVID-19 positive/negative detection.
 * [test\_COVIDx8B.txt](../labels/test_COVIDx8B.txt): This file contains the samples used for testing COVIDNet-CXR for COVID-19 positive/negative detection.

## COVIDx data distribution
COVIDx V8B
Chest radiography images distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:-----------------:|:-----------------:|:-----:|
| train |       13794       |        2158       | 15952 |
|  test |        200        |        200        |  400  |

Patients distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:-----------------:|:-----------------:|:-----:|
| train |       13651       |        1328       | 14979 |
|  test |        200        |         178       |  378  |


COVIDx V7A 
Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    5475   |   2158   | 15599 |
|  test |   100  |     100   |   200    |   400 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  7966  |    5451   |   1328   |  14745 |
|  test |   100  |     100   |    178   |    378 |
