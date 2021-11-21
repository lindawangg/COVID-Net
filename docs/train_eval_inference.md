# Training, Evaluation and Inference
## COVID-19 positive/negative detection
COVIDNet-CXR-3 model takes as input two images, one of shape (N, 256, 256, 1) for the MEDUSA architecture and one of shape (N, 480, 480, 3) for the COVIDNet architecture and outputs the softmax probabilities of COVID-19 positive and negative detection as (N, 2), where N is the number of batches.
Older COVIDNet models take a single input of shape (N, 480, 480, 3) and do not leverage a MEDUSA architecture. 

If using the TF checkpoints, here are some useful tensors:
For COVIDNet-CXR-3:
* input tensor: `input_2:0`
* input medusa tensor: `input_1:0`
* logit tensor: `final_output/MatMul:0`
* output tensor: `softmax/Softmax:0`
* label tensor: `Placeholder:0`
* class weights tensor: `Placeholder_1:0`
* loss tensor: `Mean:0`

For COVIDNet-CXR-2:
* input tensor: `input_1:0`
* logit tensor: `norm_dense_2/MatMul:0`
* output tensor: `norm_dense_2/Softmax:0`
* label tensor: `norm_dense_1_target:0`
* class weights tensor: `norm_dense_1_sample_weights:0`
* loss tensor: `Mean:0`

### Steps for training
TF training script from a pretrained model:
1. We provide you with the tensorflow evaluation script, [train_tf.py](../train_tf.py)
2. Locate the tensorflow checkpoint files (location of pretrained model)
3. To train from the COVIDNet-CXR-2 pretrained model:
```
python train_tf.py \
    --weightspath models/COVIDNet-CXR-2 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --trainfile labels/train_COVIDx9B.txt \
    --testfile labels/test_COVIDx9B.txt \
    --out_tensorname norm_dense_2/Softmax:0 \
    --logit_tensorname norm_dense_2/MatMul:0
```
4. For more options and information, `python train_tf.py --help`

### Steps for evaluation

1. We provide you with the tensorflow evaluation script, [eval.py](../eval.py)
2. Locate the tensorflow checkpoint files
3. To evaluate a tf checkpoint
For COVIDNet-CXR-3:
```
python eval.py \
    --weightspath models/COVIDNet-CXR-3 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --testfile labels/test_COVIDx9B.txt \
    --out_tensorname softmax/Softmax:0 \
    --is_medusa_backbone
```

For COVIDNet-CXR-2:
```
python eval.py \
    --weightspath models/COVIDNet-CXR-2 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --testfile labels/test_COVIDx9B.txt \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0
```
4. For more options and information, `python eval.py --help`

### Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and xray image to be inferenced
3. To inference
For COVIDNet-CXR-3:
```
python inference.py \
    --weightspath models/COVIDNet-CXR-3 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --out_tensorname softmax/Softmax:0 \
    --is_medusa_backbone
```

For COVIDNet-CXR-2:
```
python inference.py \
    --weightspath models/COVIDNet-CXR-2 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0
```
4. For more options and information, `python inference.py --help`

## Detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
COVIDNet-CXR4 models take as input an image of shape (N, 480, 480, 3) and outputs the softmax probabilities as (N, 2), where N is the number of batches.
If using the TF checkpoints, here are some useful tensors:

* input tensor: `input_1:0`
* logit tensor: `norm_dense_1/MatMul:0`
* output tensor: `norm_dense_1/Softmax:0`
* label tensor: `norm_dense_1_target:0`
* class weights tensor: `norm_dense_1_sample_weights:0`
* loss tensor: `loss/mul:0`

### Steps for training
TF training script from a pretrained model:
1. We provide you with the tensorflow evaluation script, [train_tf.py](../train_tf.py)
2. Locate the tensorflow checkpoint files (location of pretrained model)
3. To train from a pretrained model:
```
python train_tf.py \
    --weightspath models/COVIDNet-CXR4-A \
    --metaname model.meta \
    --ckptname model-18540 \
    --n_classes 3 \
    --trainfile labels/train_COVIDx8A.txt \
    --testfile labels/test_COVIDx8A.txt \
    --out_tensorname norm_dense_1/Softmax:0 \
    --logit_tensorname norm_dense_1/MatMul:0
```
4. For more options and information, `python train_tf.py --help`

### Steps for evaluation

1. We provide you with the tensorflow evaluation script, [eval.py](../eval.py)
2. Locate the tensorflow checkpoint files
3. To evaluate a tf checkpoint:
```
python eval.py \
    --weightspath models/COVIDNet-CXR4-A \
    --metaname model.meta \
    --ckptname model-18540 \
    --n_classes 3 \
    --testfile labels/test_COVIDx8A.txt \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_1/Softmax:0
```
4. For more options and information, `python eval.py --help`

### Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and xray image to be inferenced
3. To inference,
```
python inference.py \
    --weightspath models/COVIDNet-CXR4-A \
    --metaname model.meta \
    --ckptname model-18540 \
    --n_classes 3 \
    --imagepath assets/ex-covid.jpeg \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_1/Softmax:0
```
4. For more options and information, `python inference.py --help`

## Steps for Training COVIDNet-Risk

COVIDNet-Risk uses the same architecture as the existing COVIDNet - but instead it predicts the *"number of days since symptom onset"\** for a diagnosed COVID-19 patient based on their chest radiography (same data as COVIDNet). By performing offset stratification, we aim to provide an estimate of prognosis for the patient. Note that the initial dataset is fairly small at the time of writing and we hope to see more results as data increases.

1. Complete data creation and training for COVIDNet (see Training above)
2. run `train_risknet.py` (see `-h` for argument help)

*\* note that definition varies between data sources*
