# Evaluation and Inference for Pneumonia Cases

This section describes how the currently available COVIDNet chest CXR4 models can be leveraged to create a COVIDNet-P model which distinguishes between cases where
pneumonia is present and where it is not. COVIDNet-CXR4 models take as input an image of shape (N, 480, 480, 3) and output the softmax probabilities as (N, 3),
where N is the number of batches. The file inference_pneumonia.py modifies the output to return a prediction of whether
pneumonia is present or not in the given image. 

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
    --testfile test_COVIDx4.txt
    --testfolder data/test
```
4. For more options and information, `python eval_pneumonia.py --help`

## Results
These are the results generated using the eval_pneumonia.py script with the COVIDNet-CXR4-A model and test_COVIDx4.txt dataset.

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
