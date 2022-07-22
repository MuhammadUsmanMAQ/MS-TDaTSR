# Table Structure Recognition
## Usage
1. Configure `upscale_weight_dir` to the absolute path of `./MS-TDaTSR/models/upscale/` to ensure no directory errors arise while executing `upscale.py` as a subprocess. Only configure `data_root`, `work_dir` and other data related fields in the config if you want to train the model on a custom dataset.<br/>
2. Download weights from the following table and place them inside the ./MS-TDaTSR/models/struct/ directory.<br/>
#### Example usage:
```python
python inference.py --input_img "{PATH_TO_INPUT_IMG}" \
                    --upscale   # Perform AI upscaling / Use --no-upscale to disable it
                    --struct_weights "{STRUCTURE_RECOGNITION_WEIGHTS}.pth"
```

### Models Weights
_All structure recognition models have been trained and evaluated on a custom version of IBM FinTabNet Dataset._<br/>_Model Name_ (CB) represents those model weights that yield the best evaluation metrics.

| Model | Weights | Schedule |AP <sup>@ IoU=0.50</sup> | AP <sup>@ IoU=0.75</sup> | AP <sup>@ IoU=0.50:0.95</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Grid RCNN | [Download](https://drive.google.com/file/d/15xCFjnmmCj0aQlVQ6-U-bh0eICeUIWIc/view?usp=sharing) | 12 Epochs | 0.956 | 0.933 | 0.901 |

## **Directory Descriptions**

- ### Configs Directory<br/>
**Major Directory** - Contains different architectures referencing the struct_weights argument. All weights `.pth` files are named as `{arch_name}_{lr_sched}_{ckpt}.pth` and configs are thereby named as `{arch_name}.py`.<br/>
**Note** - `inference.py` automatically chooses the config that matches the struct_weight argument. If you are using a different naming convention, you will need to tinker with the source code of `inference.py` so that the appropriate config is chosen.

- ### Input Directory<br/>
**Created upon Execution** - Contains the original input image passed as a command line argument to `inference.py`. 
Useful in comparing original image with the upscaled image.

- ### Upscale Directory<br/>
**Created upon Execution** - Contains the upscaled version of the input image passed as a command line argument to `inference.py`.

- ### Output Directory<br/>
**Created upon Execution** - Stores the output image containing the bounding boxes predicted by the model. By default, the file name of the resulting image is the same as the file name of the input image.

- ### Utils Directory<br/>
Cloned from [iNNfer](https://github.com/victorca25/iNNfer). _Modified to contain specific bounding box functions to remove redundancy in the output image._

- ### Architectures Directory<br/>
Cloned from [iNNfer](https://github.com/victorca25/iNNfer). _Modified directory paths to absolute rather than relative._
