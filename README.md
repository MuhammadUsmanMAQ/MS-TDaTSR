# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields poor performance.

## Dataset
You can download and unpack the Marmot dataset (with image masks) into `datasets/stage_one` through the following link: [Marmot Dataset](https://drive.google.com/file/d/1-7cBtAraIa0e8c6kMFDPmlAlKOPOBccd/view?usp=sharing)
Alternatively, you could also download the larger cTDaR dataset (with image masks; Modern TRACK A) through the following link: [cTDaR](https://drive.google.com/file/d/1PTlz7aXY9r6sQOXApPKOyvsD6sjrjt5Q/view?usp=sharing)

## Dependencies
Install the required dependencies.<br/>Environment characteristics:<br/>`python = 3.7.13` `torch = 1.12.0` `cuda = 11.3` `torchvision = 0.13.0` `torchaudio = 0.12.0`
<br/>*It is better to create a new virtual environment so that updates/downgrades of packages do not break other projects.*
```
pip install -r requirements.txt
```
This repo uses toolboxes provided by `OpenMMLab` to train and test models. Head over to the official documentation of [MMDetection](https://github.com/open-mmlab/mmdetection) for [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Installation / Run
To get started, clone this repo and place the downloaded model weights of Upscale, Table Detection and Table Structure Recognition in the models directory.<br/>
**Note: _You will need to configure weight directories in specific .py files in libs_det/ & libs_struct/ if you want to execute scripts without command line arguments._** <br/>
Your directory should look something like as the following tree:
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

1. Download the model weights provided in the [Table Detection](libs_det#table-detection) as `det_weights` and [Table Structure Recognition](libs_struct#table-structure-recognition) as `struct_weights`.
2. Either configure `det_weights`, `struct_weights`, `input_dir` and `output_dir` in `run.py` or pass in command line arguments. Example usage:
```python
python run.py --input_dir "{PATH_TO_INPUT_IMAGE}" \
              --det_weights "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
              --struct_weights "{STRUCTURE_DETECTOR_WEIGHTS}.pth.tar" \
              --output_dir "{PATH_TO_OUTPUT_IMAGE}"
```
<p align="center">
    <p1 align="center"> <b>Note:</b> <i>Individual inference (performing either table detection or structure extraction) is also possible.</i>
</p>

## Issues
- Machines running variants of Microsoft Windows encounter issues with mmcv imports. Follow the [installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) on the official MMCV documentation to resolve such issues. Example:
```
ModuleNotFoundError: No module named 'mmcv._ext'
```
- Machinces running variants of Microsoft Windows encounter directory issues arising from OSP. Most can be resolved by using absolute path in the command line arguments rather than the relative path.



