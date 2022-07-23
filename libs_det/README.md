# Table Detection
## Usage
1. Configure `base_dir` and `data_dir` in `configs/directories.py` to the absolute paths to ensure no directory errors arise during training/testing.
2. Run `data.py` to ensure data is being loaded correctly.
2. Configure `encoder/decoder` in `configs/config.py` to match the downloaded weights file.<br/>Example: `TD-ConvNeXt-T` implies `Encoder: ConvNeXt` and `CNDecoder`.
3. To train the model, configure the training hyperparameters in `configs/config.py` and execute `train.py`.
4. Download weights from the following table and place them inside the ./MS-TDaTSR/models/det/ directory
5. To evaluate the model on a test dataset, use the script `eval.py` with the required positional arguments.
### Example usage:
```python
python eval.py --output_dir "{PATH_TO_OUTPUT_DIRECTORY}" \
               --det_weights "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
               --data_dir "{PATH_TO_INPUT_DIR}"
```
5. To get model predictions for a single input image, use the script `inference.py` with the required positional arguments. Example usage:
```python
python inference.py --input_img "{PATH_TO_INPUT_IMG}" \
                    --gt_dir "{PATH_TO_INPUT_MASK}" \ # Optional
                    --det_weights "{TABLE_DETECTOR_WEIGHTS}.pth.tar" \
                    --output_dir "{{PATH_TO_OUTPUT_DIRECTORY}"
```
## Models Weights
_All table detection models have been trained and evaluated on cTDaR Modern TRACK A Dataset._<br/>_Model Name_ (CB) represents those model weights that yield the best evaluation metrics.

| Model | Weights | Schedule |AP <sup>@ IoU=0.50</sup> | AP <sup>@ IoU=0.75</sup> | AP <sup>@ IoU=0.50:0.95</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| TD-ConvNeXt-T | [Download](https://drive.google.com/file/d/1-INWLZ8RPdEM5mpqASL7Mx3dvVvpktw_/view?usp=sharing) | 30 Epochs | 0.958 | 0.953 | 0.954 |
| **TD-ConvNeXt-S** (CB) | [Download](https://drive.google.com/file/d/1-A0W1Z0YNWifHLkCDbOutMqSVgz4PRPN/view?usp=sharing) | 30 Epochs | 0.988 | 0.984 | 0.984 |
| TD-EfficientNet-B3 | [Download](https://drive.google.com/file/d/1-2F-DMPX2IL2PMnZxTK_2BkLnRNuF0P9/view?usp=sharing) | 30 Epochs | 0.977 | 0.974 | 0.971 |
