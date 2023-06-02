# Object-detection
project for object detection

# Before running
Download from Kaggle yolov3 model: https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?resource=download

```sh
mkdirs models/yolov3
```
put models and classes into the directory

# Build
Launch script settings.sh to configure requirements and dependencies

# Command line
Default it use the yolov3 model

```sh
$ python main.py --input <path_input> 
```
if it needs is possibile to use specific model with relative classes

```sh
$ python3 main.py --input <path_input> --model <model_input> --class <class_input> 
```
# Function
- search : add string or index class that want to find during object detection
- input : Can be folder, single image or a video file
- create : create a video file with detected class

# TODO 

- Set virtualenv
