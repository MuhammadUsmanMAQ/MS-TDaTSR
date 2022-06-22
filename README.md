# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields poor performance.

## Dataset
You can download and unpack the Marmot dataset (with image masks) into `datasets/stage_one` through the following link: [Marmot Dataset](https://drive.google.com/file/d/1-7cBtAraIa0e8c6kMFDPmlAlKOPOBccd/view?usp=sharing)
Alternatively, you could also download the larger cTDaR dataset (with image masks) through the following link: [cTDaR](https://drive.google.com/file/d/1HmIISGpIdewxenQCaqcUi1bQNfiFNhxr/view?usp=sharing)

## Installation / Custom Runs
Install the required dependencies.<br/>Environment characteristics: `python = 3.9.0` `torch = 1.10.0` `torchvision = 0.11.0` `torchaudio = 0.10.0`
<br/>*It is better to create a new virtual environment so that updates/downgrades of packages do not break other projects.*
```
pip install -r requirements.txt
```
## Models
Download pretrained model weights through the following tables. _More models are added as soon as they are trained/tested._
### **Trained on Marmot Dataset**
| Model | Weights | F1-Score | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: |
| | | | | |

### **Trained on cTDaR Dataset**

| Model | Weights | F1-Score | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: |
| ConvNeXt-Tiny-CNDecoder | [Download](https://drive.google.com/file/d/1-INWLZ8RPdEM5mpqASL7Mx3dvVvpktw_/view?usp=sharing) | 0.954 | 0.960 | 0.954 |

## Usage
To get started, either clone this repo or arrange your working directory as:
```
MS-TDaSR/
├── datasets
│   └── stage_one
│       ├── images/..
│       ├── table_mask/..
│       └── locate.csv
├── libs
│   └── stage_one
│       ├── config.py
│       ├── data.py
│       ├── decoder.py
│       ├── encoder.py
│       ├── eval.py
│       ├── loss.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── models
│   └── stage_one
│       ├── EfficientNet-B4-Decoder/..
│       ├── ResNet-101_RNDecoder/..
│       └── /...
├── outputs
│   └── stage_one
│       └── metrics.csv
└── requirements.txt
```
- **Stage 1**
1. Configure `base_dir` and `data_dir` in `config.py` and run `data.py` to ensure data is being loaded correctly.
2. To make changes to the model _(i.e. changing the encoder, using ConvNeXt instead of ResNet, etc.)_, configure `encoder/decoder` in `config.py`.
3. To train the model, configure the training hyperparameters in `config.py` and run `train.py`.
4. To evaluate the model on a test dataset, use the script `eval.py` with the required positional arguments.
Example usage:
```
python eval.py --output_dir "{config.base_dir}/outputs/stage_one/{config.encoder}_{config.decoder}" \
               --model_dir "{config.base_dir}/models/stage_one/{config.encoder}_{config.decoder}/{model_name}.pth.tar" \
               --data_dir "{config.data_dir}"
```
5. To get model predictions for a single input image, use the script `inference.py` with the required positional arguments. Example usage:
```
python inference.py --input_img "{path_to_input}" \
                    --gt_dir "{path_to_gt}" \ # Optional
                    --model_dir "{config.base_dir}/models/stage_one/{config.encoder}_{config.decoder}/{model_name}.pth.tar" \
                    --output_dir "{config.base_dir}/outputs/stage_one/{config.encoder}_{config.decoder}"
```
