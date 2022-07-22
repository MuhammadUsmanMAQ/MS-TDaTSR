<p align="center">
   <a href="#"><img width="160" height="25" src="./resources/status-in progress-critical.svg"/></a>
   <a href="#"><img width="105" height="25" src="./resources/unfunctional-blue.svg"/></a><br/>
</p>

# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields relatively poor performance.

<p align="center">
   <a href="https://pytorch.org/"><img width="95" height="25" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white"/></a>
   <!--- <a href="https://pandas.pydata.org/"><img width="90" height="25" src="https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white"/></a> -->
   <a href="https://numpy.org/"><img width="90" height="25" src="https://img.shields.io/badge/OpenCV-27338e?flat&logo=OpenCV&logoColor=white"/></a>
   <a href="https://numpy.org/"><img width="90" height="25" src="https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white"/></a>
   <a href="https://www.python.org/"><img width="90" height="25" src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54"/></a>
   <a href="#"><img width="155" height="25" img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Demo"/></a>
   <a href="#"><img width="150" height="25" img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Demo"/></a>
</p>

## Datasets
- **Table Detection** - You can download and unpack the Marmot dataset (with image masks) into `datasets/stage_one` through the following link: [Marmot Dataset](https://drive.google.com/file/d/1-7cBtAraIa0e8c6kMFDPmlAlKOPOBccd/view?usp=sharing)
Alternatively, you could also download the larger cTDaR dataset (with image masks; Modern TRACK A) through the following link: [cTDaR](https://drive.google.com/file/d/1PTlz7aXY9r6sQOXApPKOyvsD6sjrjt5Q/view?usp=sharing)<br/>
- **Table Structure Recognition** - You can access download links to [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/) from the official IBM developer website.
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

1. Download the model weights provided in the [Detection Weights](libs_det#models-weights) for any `det_weights` argument and [Structure Recognition Weights](libs_struct#models-weights) for any `struct_weights` argument.
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

## Acknowledgements
**Special thanks to the following contributors without which this repo would not be possible:**
1. The [MMDetection](https://github.com/open-mmlab/mmdetection) team for creating their amazing framework to push the state of the art computer vision research and enabling us to experiment and build various models very easily.
<p align="center">
   <a href="https://github.com/open-mmlab/mmdetection"><img width="300" height="100" src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/mmdet-logo.png"/></a>
</p>

2. The [CRAFT](https://github.com/clovaai/CRAFT-pytorch) paper implementation which enabled us to perform fast and lite text detection during post-processing.
3. The [GameUpscale](https://upscale.wiki/wiki/Main_Page) team for providing a plethora of models to upscale all kinds of images; upscaling text images for our case.
<p align="center">
   <a href="#"><img width="100" height="100" src="https://styles.redditmedia.com/t5_t2w6c/styles/communityIcon_lslg93wlmah31.png?width=256&s=3163c0903846807d8609680be18368a0a7eef05b"/></a>
</p>
   
4. [Google Colaboratory](https://github.com/googlecolab) for providing free high end GPU resources for research and development. All of the code base was developed using their platform and could not be possible without it.
