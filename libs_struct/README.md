# Structure Recognition Libraries
Configure `base_dir` to the absolute path of `./MS-TDaTSR` to ensure no directory errors arise while executing `inference.py`. Only configure `data_root`, `work_dir` and other data related fields in the config if you want to train the model on a custom dataset.<br/>
## **Directory Descriptions**

### Input Directory<br/>
Contains the original input image passed as a command line argument to `inference.py`. 
Useful in comparing original image with the upscaled image.

### Upscale Directory<br/>
Contains the upscaled version of the input image passed as a command line argument to `inference.py`.

### Output Directory<br/>
Stores the output image containing the bounding boxes predicted by the model. By default, the file name of the resulting image is the same as the file name of the input image.

### Utils Directory<br/>
Cloned from [iNNfer](https://github.com/victorca25/iNNfer). _Modified to contain specific bounding box functions to remove redundancy in the output image._

### Architectures Directory<br/>
Cloned from [iNNfer](https://github.com/victorca25/iNNfer). _Modified directory paths to absolute rather than relative._
