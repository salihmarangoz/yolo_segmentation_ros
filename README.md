# yolo_segmentation_ros

Sweet pepper detection+segmentation+tracking via YOLOv8. Alternative to `agrobot_mrcnn_ros` and `mask_rcnn_ros`. Uses `mask_rcnn_ros/Result` for publishing the results.

Watch my quick experiment on YouTube:

[![Watch the video](https://img.youtube.com/vi/-oo29vo7gBQ/maxresdefault.jpg)](https://youtu.be/-oo29vo7gBQ)

## Important Information!

[ultralytics](https://github.com/ultralytics/ultralytics) is a nice YOLOv8 package, but it comes with [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) which allows for personal and academic use **but not for commercial use!** If you would like to improve this package's license restrictions and performance, replace this library with [super-gradients](https://github.com/Deci-AI/super-gradients) which has YOLOv8-NAS outperforming ultralytics's YOLOv8. There are pretty good examples on its [Github Readme](https://github.com/Deci-AI/super-gradients) and [here](https://www.kaggle.com/general/406701).

## Installation

```bash
$ pip install -r requirements.txt
```

**Download Model:** If you would like to skip the training part, download the `best.pt` from [HERE](https://drive.google.com/drive/folders/1aZ4MpL7zXARpdr7hky6iucenSEoaDH9W?usp=sharing) and place it into the `model` folder. Click [here](https://docs.ultralytics.com/tasks/segment/#models) for more information about the base models.

Tested on;

- Ubuntu 20.04 + ROS Noetic
- Ryzen 5900HX + RTX3080M 16GB
- Nvidia Driver 525.105.17
- CUDA 11.7
- **ultralytics 8.0.114**

## Training

 ```bash
 $ cd yolo_segmentation_ros/train
 ```

1. **(Mandatory)** Prepare BUP20
   1. Download [BUP20](http://agrobotics.uni-bonn.de/sweet_pepper_dataset/) dataset and extract `BUP20.tar.gz` into a folder. (it should have the name `CKA_sweet_pepper_2020_summer`)
   2. Modify `DATASET_PATH` at the top of the notebook then run the notebook `train/bup20_to_yolov8.ipynb ` to create yolov8 dataset. 

2. **(Optional)** Prepare Sweet Pepper Dataset-1 from Kaggle (nameless):
   1. Download [Sweet Pepper Dataset from Kaggle](https://www.kaggle.com/datasets/lemontyc/sweet-pepper?resource=download) and extract into a folder. (it should have the name `dataset_620_red_yellow_cart_only`)
   2. Modify `DATASET_PATH` at the top of the notebook then run the notebook `train/dataset_620_to_yolov8.ipynb ` to create yolov8 dataset.

3. **(Mandatory)** Split train/validation using `split_train_val.ipynb`

After these steps, the dataset should be ready. For training the model:

1. Run the script `train/train.py` for training. This should take about 15-20 mins.
2. Copy the `train/runs/segment/train/weights/best.pt` to `model/best.pt`

## Prediction

```bash
$ cd yolo_segmentation_ros/predict
```

- Predict a region of your screen:

  ```bash
  $ python predict_screen.py
  ```

- Predict using your web-cam:

  ```bash
  $ python predict_webcam.py
  ```

- Predict using a video:

  ```bash
  $ python predict_video.py.py
  ```

- Predict using images:

  ```bash
  $ python predict_images.py
  ```


## ROS Node

The ROS node script is located here: `scripts/ros_node.py`

TODO



## To-Do

- [BUP19](http://agrobotics.uni-bonn.de/sweet_pepper_dataset/) has bitmap masks which needs to be processed to extract polygons. It would be a nice improvement if we can add this to the training. See https://github.com/open-mmlab/mmdetection/issues/7816
