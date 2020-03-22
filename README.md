# COVID-Net and COVIDx Dataset
<p align="center">
	<img src="assets/covid-2p-rca.png" alt="photo not available" width="70%" height="70%">
	<br>
	<em>Example chest radiography images of COVID-19 cases from 2 different patients and their associated critical factors (highlighted in red) as identified by GSInquire.</em>
</p>

[Linda Wang and Alexander Wong, "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images", 2020. (pdf)](assets/COVID_Net.pdf)

The COVID-19 pandemic continues to have a devastating effect on the health and well-being of global population. A critical step in the fight against COVID-19 is effective screening of infected patients, with one of the key screening approaches being radiological imaging using chest radiography. It was found in early studies that patients present abnormalities in chest radiography images that are characteristic of those infected with COVID-19.  Motivated by this, a number of artificial intelligence (AI) systems based on deep learning have been proposed and results have been shown to be quite promising in terms of accuracy in detecting patients infected with COVID-19 using chest radiography images. However, to the best of the authors' knowledge, these developed AI systems have been closed source and unavailable to the research community for deeper understanding and extension, and unavailable for public access and use. Therefore, in this study we introduce COVID-Net, a deep convolutional neural network design tailored for the detection of COVID-19 cases from chest radiography images that is open source and available to the general public. We also describe the chest radiography dataset leveraged to train COVID-Net, which we will refer to as COVIDx and is comprised of 5941 posteroanterior chest radiography images across 2839 patient cases from two open access data repositories. Furthermore, we investigate how COVID-Net makes predictions using an explainability method in an attempt to gain deeper insights into critical factors associated with COVID cases, which can aid clinicians in improved screening. By no means a production-ready solution, the hope is that the open access COVID-Net, along with the description on constructing the open source COVIDx dataset, will be leveraged and build upon by both researchers and citizen data scientists alike to accelerate the development of highly accuracy yet practical deep learning solutions for detecting COVID-19 cases and accelerate treatment of those who need it the most.

If you would like to contribute COVID-19 x-ray images, please contact us at linda.wang513@gmail.com and a28wong@uwaterloo.ca/alex@darwinai.ca. Lets all work together to stop the spread of COVID-19!

Our desire is to encourage broad adoption and contribution to this project. Accordingly this project has been licensed under the GNU Affero General Public License 3.0. Please see [license file](LICENSE.md) for terms. If you would like to discuss alternative licensing models, please reach out to us at: linda.wang513@gmail.com and a28wong@uwaterloo.ca or alex@darwinai.ca.

## COVIDx Dataset
Currently, the COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

The COVIDx dataset can be downloaded [here](https://drive.google.com/file/d/1-T26bHP7MCwB8vWeKufjGmPKl8pesM1J/view?usp=sharing).
Preprocessed ready-for-training COVIDx dataset can be downloaded [here](https://drive.google.com/file/d/1zCnmcMxSRZTqJywur7jCqZk0z__Mevxp/view?usp=sharing).

Chest radiography images distribution
|  Type | Normal | Viral | Bacteria | COVID-19 | Total |
|:-----:|:------:|:-----:|:--------:|:--------:|:-----:|
| train |  1349  |  1355 |   2540   |    60    |  5304 |
|  test |   234  |  149  |    246   |     8    |   637 |

Patients distribution
|  Type | Normal | Viral | Bacteria | COVID-19 | Total |
|:-----:|:------:|:-----:|:--------:|:--------:|:-----:|
| train |  1001  |  534  |   853    |    41    | 2429  |
|  test |   202  |  126  |    78    |     4    |  410  |

## Training and Evaluation
(Releasing soon)

## Results
<p align="center">
	<img src="assets/confusion.png" alt="photo not available" width="70%" height="70%">
	<br>
	<em>Confusion matrix for COVID-Net on the COVIDx test dataset.</em>
</p>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="4">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Bacterial</td>
    <td class="tg-7btt">Non-COVID19 Viral</td>
    <td class="tg-7btt">COVID-19 Viral</td>
  </tr>
  <tr>
    <td class="tg-c3ow">73.9</td>
    <td class="tg-c3ow">93.1</td>
    <td class="tg-c3ow">81.9</td>
    <td class="tg-c3ow">100.0</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="4">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Bacterial</td>
    <td class="tg-7btt">Non-COVID19 Viral</td>
    <td class="tg-7btt">COVID-19 Viral</td>
  </tr>
  <tr>
    <td class="tg-c3ow">95.1</td>
    <td class="tg-c3ow">87.1</td>
    <td class="tg-c3ow">67.0</td>
    <td class="tg-c3ow">80.0</td>
  </tr>
</table></div>

## Pretrained Models
(Releasing soon)
