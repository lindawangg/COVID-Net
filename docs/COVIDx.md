# COVIDx Dataset
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

<!--We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Actualmed, M.E.H. Chowdhury et al., Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.-->

## Steps to generate the dataset

1. Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git`
 * go to this [link](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) to download the COVID-19 Radiography database. Only the COVID-19 image folder and metadata file is required. The overlaps between covid-chestxray-dataset are handled
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [create\_COVIDx\_binary.ipynb](../create_COVIDx_binary.ipynb) to combine the three datasets to create COVIDx for binary classification. Make sure to remember to change the file paths. Use [create\_COVIDx.ipynb](../create_COVIDx.ipynb) for datasets compatible with COVIDx5 and earlier models (not binary classification).
4. We provide the train and test txt files with patientId, image path and label. Note that the label is 'positive' or 'negative' for COVIDx6 and later or 'normal', 'pneumonia', and 'COVID-19' for COVIDx5 and earlier. The description for each file is explained below:
 * [train\_COVIDx5.txt](../labels/train_COVIDx5.txt): This file contains the samples used for training COVIDNet-CXR for multi-class classification.
 * [test\_COVIDx5.txt](../labels/test_COVIDx5.txt): This file contains the samples used for testing COVIDNet-CXR for multi-class classification.
 * [train\_COVIDx6.txt](../labels/train_COVIDx6.txt): This file contains the samples used for training COVIDNet-CXR for binary classification.
 * [test\_COVIDx6.txt](../labels/test_COVIDx6.txt): This file contains the samples used for testing COVIDNet-CXR for binary classification..

## COVIDx data distribution
V6 Binary Distribution
Chest radiography images distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:------:|:---------:|:--------:|
| train |  13794  |    517   |    14311   |
|  test |   100  |     100   |   200    |

Patients distribution
|  Type | COVID-19 Negative | COVID-19 Positive | Total |
|:-----:|:------:|:---------:|:--------:|
| train |  13671  |    353   |    14024   |
|  test |   100  |      74   |     174   |


V5 Case Distribution
Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    5475   |   517    | 13958 |
|  test |   100  |     100   |   100    |   300 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  7966  |    5451   |    353   |  13770 |
|  test |   100  |      98   |     74   |    272 |
