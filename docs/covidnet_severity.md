# COVIDNet Lung Severity Scoring
COVIDNet-S-GEO and COVIDNet-S-OPC models takes as input a chest x-ray image of shape (N, 480, 480, 3), where N is the number of batches, and outputs the SARS-CoV-2 severity scores for geographic extent and opacity extent, respectively. COVIDNet-S-GEO predicts the geographic severity. Geographic severity is based on the geographic extent score for right and left lung. For each lung: 0 = no involvement; 1 = <25%; 2 = 25-50%; 3 = 50-75%; 4 = >75% involvement, resulting in scores from 0 to 8. COVIDNet-S-OPC predicts the opacity severity. Opacity severity is based on the opacity extent score for right and left lung. For each lung, the score is from 0 to 4, with 0 = no opacity and 4 = white-out, resulting in scores from 0 to 8. For detailed description of COVIDNet lung severity scoring methodology, see the paper [here](https://arxiv.org/abs/2005.12855).

If using the TF checkpoints, here are some useful tensors:

* input tensor: `input_1:0`
* logit tensor: `MLP/dense_1/MatMul:0`
* is_training tensor: `keras_learning_phase:0`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download the COVIDNet-S Lung Severity Scoring models from the [pretrained models section](models.md)
2. Locate both geographic and opacity models and COVID-19 positive chest x-ray image to be inferenced
3. To predict geographic and opacity severity
```
python inference_severity.py \
    --weightspath_geo models/COVIDNet-S-GEO \
    --weightspath_opc models/COVIDNet-S-OPC \
    --metaname model.meta \
    --ckptname model \
    --imagepath assets/ex-covid.jpeg
```
4. For more options and information, `python inference_severity.py --help`
