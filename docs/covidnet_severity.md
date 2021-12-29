# COVIDNet CXR-S Air Space Severity Grading
COVIDNet CXR-S model takes as input a chest x-ray image of shape (N, 480, 480, 3). where N is the number of batches, 
and outputs the airspace severity of a SARS-CoV-2 positive patient. The airspace severity is grouped into two levels: 1) Level 1: opacities in 1-2 lung zones, and 2) Level 2: opacities in 3 or more lung zones.

For a detailed description on the methodology behind COVIDNet CXR-S, please click [here](https://www.mdpi.com/2075-4418/12/1/25).

If using the TF checkpoints, here are some useful tensors:

* input tensor: `input_1:0`
* logit tensor: `norm_dense_2/MatMul:0`
* output tensor: `norm_dense_2/Softmax:0`
* label tensor: `norm_dense_1_target:0`
* class weights tensor: `norm_dense_1_sample_weights:0`
* loss tensor: `Mean:0`

### Steps for training
To train the model the COVIDxSev dataset is required, to create the dataset please run [create_COVIDxSev.ipynb](../create_COVIDxSev.ipynb).
TF training script from a pretrained model:
1. We provide you with the tensorflow training script, [train_tf.py](../train_tf.py)
2. Locate the tensorflow checkpoint files (location of pretrained model)
3. To train from the COVIDNet-CXR-S pretrained model:
```
python train_tf.py \
    --weightspath models/COVIDNet-CXR-S \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --datadir data_sev \
    --trainfile labels/train_COVIDxSev.txt \
    --testfile labels/test_COVIDxSev.txt \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0 \
    --logit_tensorname norm_dense_2/MatMul:0 \
    --is_severity_model
```
4. For more options and information, `python train_tf.py --help`

### Steps for evaluation
To evaluate the model the COVIDxSev dataset is required, to create the dataset please run [create_COVIDxSev.ipynb](../create_COVIDxSev.ipynb).
1. We provide you with the tensorflow evaluation script, [eval.py](../eval.py)
2. Locate the tensorflow checkpoint files
3. To evaluate a tf checkpoint:
```
python eval.py \
    --weightspath models/COVIDNet-CXR-S \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --testfolder data_sev/test \
    --testfile labels/test_COVIDxSev.txt \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0 \
    --is_severity_model
```
4. For more options and information, `python eval.py --help`

### Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and xray image to be inferenced
3. To inference,
```
python inference.py \
    --weightspath models/COVIDNet-CXR-S \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0 \
    --is_severity_model
```
4. For more options and information, `python inference.py --help`

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
