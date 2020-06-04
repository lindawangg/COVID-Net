# COVIDNet Lung Severity Scoring
COVIDNet-SEV-GEO and COVIDNet-SEV-OPC models takes as input a chest x-ray image of shape (N, 480, 480, 3), where N is the number of batches, and outputs the SARS-CoV-2 severity scores for geographic extent and opacity extent, respectively. COVIDNet-SEV-GEO predicts the geographic severity. Geographic severity is based on the geographic extent score for right and left lung. For each lung: 0 = no involvement; 1 = <25%; 2 = 25-50%; 3 = 50-75%; 4 = >75% involvement, resulting in scores from 0 to 8. COVIDNet-SEV-OPC predicts the opacity severity. Opacity severity is based on the opacity extent score for right and left lung. For each lung: 0 = no opacity; 1 = ground glass opacity; 2 =consolidation; 3 = white-out, resulting in scores from 0 to 6. For detailed description of COVIDNet lung severity scoring methodology, see the paper [here](https://arxiv.org/abs/2005.12855).

If using the TF checkpoints, here are some useful tensors:

* input tensor: `input_1:0`
* logit tensor: `MLP/dense_1/MatMul:0`
* is_training tensor: `keras_learning_phase:0`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download the COVIDNet Lung Severity Scoring models from the [pretrained models section](models.md)
2. Locate both geographic and opacity models and COVID-19 positive chest x-ray image to be inferenced
3. To predict geographic and opacity severity
```
python inference_severity.py \
    --weightspath_geo models/COVIDNet-SEV-GEO \
    --weightspath_opc models/COVIDNet-SEV-OPC \
    --metaname model.meta \
    --ckptname model \
    --imagepath assets/ex-covid.jpeg
```
4. For more options and information, `python inference_severity.py --help`
