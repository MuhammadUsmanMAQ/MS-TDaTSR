# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields poor performance.

## Installation / Custom Runs
Begin with installing required packages.<br/>*It is better to create a new environment so that this repo does not interfere with other projects.*
```
pip install -r requirements.txt
```
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
