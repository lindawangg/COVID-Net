# Inference for Pneumonia Cases

COVIDNet-CXR4 models takes as input an image of shape (N, 480, 480, 3) and outputs the softmax probabilities as (N, 3),
where N is the number of batches. The file inference_pneumonia.py then modifies the output to return a prediction of whether
pneumonia is present or not.

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
