# COVIDx Dataset
**Update 04/15/2020: Released new dataset with 152 COVID-19 train and 31 COVID-19 test samples. There are constantly new xray images being added to covid-chestxray-dataset and Figure1 covid dataset so we included train_COVIDx2.txt and test_COVIDx2.txt, which are the xray images we used for training and testing of the CovidNet-CXR models.**

The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.

## Steps to generate the dataset

1. Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * `git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset`
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [create\_COVIDx\_v3.ipynb](../create_COVIDx_v3.ipynb) to combine the three dataset to create COVIDx. Make sure to remember to change the file paths.
4. We provide the train and test txt files with patientId, image path and label (normal, pneumonia or COVID-19). The description for each file is explained below:
 * [train\_COVIDx2.txt](../train_COVIDx2.txt): This file contains the samples used for training COVIDNet-CXR.
 * [test\_COVIDx2.txt](../test_COVIDx2.txt): This file contains the samples used for testing COVIDNet-CXR.

## COVIDx data distribution

Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    5451   |   152    | 13569 |
|  test |   100  |     100   |    31    |   231 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  7966  |    5440   |    107   |  13513 |
|  test |   100  |      98   |     14   |    212 |
