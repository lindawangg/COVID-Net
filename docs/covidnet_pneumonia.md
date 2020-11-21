# COVIDNet-P: COVID-Net for Pneumonia Detection

This section describes how we leveraged COVID-Net to create a COVIDNet-P model which distinguishes between cases where pneumonia is present and where it is not. Using the inference_pneumonia.py file provided, COVIDNet-P takes as input an image of shape (N, 480, 480, 3) and outputs the likelihood of whether pneumonia is present or not in the given image.

## Steps for Inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and xray image to be inferenced
3. To inference,
```
python inference_pneumonia.py \
    --weightspath models/COVIDNet-CXR4-A \
    --metaname model.meta \
    --ckptname model-18540 \
    --imagepath assets/ex-covid.jpeg
```
4. For more options and information, `python inference_pneumonia.py --help`

## Steps for Evaluation

1. Download a model from the [pretrained models section](models.md)
2. Download a test dataset from the [main repo](https://github.com/lindawangg/COVID-Net)
3. To evaluate,
```
python eval_pneumonia.py \
    --weightspath models/COVIDNet-CXR4-A \
    --metaname model.meta \
    --ckptname model-18540 \
    --testfile test_COVIDx5.txt
    --testfolder data/test
```
4. For more options and information, `python eval_pneumonia.py --help`

## Results
These are the results generated using the eval_pneumonia.py script with the COVIDNet-CXR4-A model acting as the backbone of COVIDNet-P and test_COVIDx5.txt dataset

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
  </tr>
  <tr>
    <td class="tg-c3ow">94.0</td>
    <td class="tg-c3ow">95.0</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
  </tr>
  <tr>
    <td class="tg-c3ow">90.4</td>
    <td class="tg-c3ow">96.9</td>
  </tr>
</table></div>
