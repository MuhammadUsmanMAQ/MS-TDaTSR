# MS-TDaTSR
## Mutli-Stage - Table Detection and Table Structure Recognition
We propose a multi-staged pipeline approach to tackle the issue of detection of tables as well as their structure within scanned document images, on the basis that the fundamental structure of bordered and borderless tables is vastly different and hence, training a single pipeline model to discern the structure of both borderless and bordered tables yields poor performance.

## Installation / Custom Runs
Begin with installing required packages.<br/>*It is better to create a new environment so that this repo does not interfere with other projects.*
```
pip install -r requirements.txt
```
- **For Stage 1**
1. Configure **_DIR_TO_BE_CONFIGURED_** in `config.py, data.py, train.py` and execute the scripts.
2. Configure **_TEST_DIR_** and **_OUTPUT_DIR_** in eval.py to get metrics and output image masks/bounding boxes.
3. To make changes to the model _(i.e. changing the encoder, using ConvNeXt instead of ResNet, etc.)_, configure the respective files in `libs` directory.
