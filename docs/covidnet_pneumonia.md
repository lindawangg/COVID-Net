# Inference for Pneumonia Cases

This section describes how the currently available models are able to be leveraged for use distinguishing only between cases where
pneumonia is present and where it is not. The file inference_pneumonia.py makes use of COVIDNet-CXR models and modifies the output to return a prediction of whether
pneumonia is present or not in a given image.

## Steps for inference
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
