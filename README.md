# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields poor performance.

## Dataset
You can download and unpack the Marmot dataset (with image masks) into `datasets/stage_one` through the following link: [Marmot Dataset](https://drive.google.com/file/d/1-7cBtAraIa0e8c6kMFDPmlAlKOPOBccd/view?usp=sharing)
Alternatively, you could also download the larger cTDaR dataset (with image masks; Modern TRACK A) through the following link: [cTDaR](https://drive.google.com/file/d/1PTlz7aXY9r6sQOXApPKOyvsD6sjrjt5Q/view?usp=sharing)

## Dependencies
Install the required dependencies.<br/>Environment characteristics:<br/>`python = 3.7.13` `torch = 1.11.0` `cuda = 11.3` `torchvision = 0.12.0` `torchaudio = 0.11.0`
<br/>*It is better to create a new virtual environment so that updates/downgrades of packages do not break other projects.*
```
pip install -r requirements.txt
```

## Installation / Run
To get started, clone this repo and place the downloaded model weights of Upscale, Table Detection and Table Structure Recognition in the models directory.<br/>
Note: _You will need to configure weight directories in the libs_det & libs_struct if you want to run .py files without command line arguments.
```python
MS-TDaSR/
├── datasets
│   └── /...
├── libs-det
│   ├── config.py
│   ├── data.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── eval.py
│   ├── inference.py
│   ├── inference_batch.py
│   ├── loss.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── libs-struct
│   ├── architectures/
│   ├── utils/
│   ├── config.py
│   ├── config.py
│   └── inference.py
├── models
│   ├── det/(Downloaded Weights.pth)
│   ├── struct/(Downloaded Weights.pth)
│   └── upscale/(Downloaded Weights.pth)
├── run.py
└── requirements.txt
```

1. Download the model weights provided in the [Table Detection](#table-detection) as `det_weights` and [Table Structure Recognition](#table-structure-recognition) as `struct_weights`.
2. Either configure `det_weights`, `struct_weights`, `input_dir` and `output_dir` in `run.py` or pass in command line arguments. Example usage:
```python
python run.py --input_dir "{PATH_TO_INPUT_IMAGE}" \
              --det_weights "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
              --struct_weights "{STRUCTURE_DETECTOR_WEIGHTS}.pth.tar" \
              --output_dir "{PATH_TO_OUTPUT_IMAGE}"
```
<p align="center">
    <p1 align="center"> <b>Note:</b> <i>Individual inference (performing either table detection or structure extraction) is also possible.</i>
    <br>
    <b>We also provide pretrained weights along with their schedule details in the sections that follow.</b> </p1>
</p>

## Table Detection
### Models Weights
_All table detection models have been trained and evaluated on cTDaR Modern TRACK A Dataset._<br/>_Model Name_ (CB) represents those model weights that yield the best evaluation metrics.

| Model | Weights | Schedule |AP <sup>@ IoU=0.50</sup> | AP <sup>@ IoU=0.75</sup> | AP <sup>@ IoU=0.50:0.95</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| TD-ConvNeXt-T | [Download](https://drive.google.com/file/d/1-INWLZ8RPdEM5mpqASL7Mx3dvVvpktw_/view?usp=sharing) | 30 Epochs | 0.958 | 0.953 | 0.954 |
| **TD-ConvNeXt-S** (CB) | [Download](https://drive.google.com/file/d/1-A0W1Z0YNWifHLkCDbOutMqSVgz4PRPN/view?usp=sharing) | 30 Epochs | 0.988 | 0.984 | 0.984 |
| TD-EfficientNet-B3 | [Download](https://drive.google.com/file/d/1-2F-DMPX2IL2PMnZxTK_2BkLnRNuF0P9/view?usp=sharing) | 30 Epochs | 0.977 | 0.974 | 0.971 |

### Usage
1. Configure `base_dir` and `data_dir` in `config.py` and run `data.py` to ensure data is being loaded correctly.
2. To make changes to the model _(i.e. changing the encoder, using ConvNeXt instead of ResNet, etc.)_, configure `encoder/decoder` in `config.py`.
3. To train the model, configure the training hyperparameters in `config.py` and run `train.py`.
4. To evaluate the model on a test dataset, use the script `eval.py` with the required positional arguments.
Example usage:
```python
python eval.py --output_dir "{PATH_TO_OUTPUT_DIRECTORY}" \
               --model_dir "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
               --data_dir "{PATH_TO_INPUT_DIR}"
```
5. To get model predictions for a single input image, use the script `inference.py` with the required positional arguments. Example usage:
```python
python inference.py --input_img "{PATH_TO_INPUT_IMG}" \
                    --gt_dir "{PATH_TO_INPUT_MASK}" \ # Optional
                    --model_dir "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
                    --output_dir "{{PATH_TO_OUTPUT_DIRECTORY}"
```

## Table Structure Recognition
`Status: In Progress`
<!---
**Evaluated on _**

| Model | Weights | Schedule |AP <sup>@ IoU=0.50</sup> | AP <sup>@ IoU=0.75</sup> | AP <sup>@ IoU=0.50:0.95</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| | [Download]() | | | | |
-->
