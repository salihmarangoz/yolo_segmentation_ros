# yolo_segmentation_ros

Sweet pepper detection+segmentation+tracking via YOLOv8. Alternative to `agrobot_mrcnn_ros` and `mask_rcnn_ros`. Uses `mask_rcnn_ros/Result` for publishing the results.

## Important Information

[ultralytics](https://github.com/ultralytics/ultralytics) is a nice YOLOv8 package, but it comes with [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) which allows for personal and academic use **but not for commercial use!** If you would like to improve this package's license restrictions and performance, replace this library with [super-gradients](https://github.com/Deci-AI/super-gradients) which has YOLOv8-NAS outperforming ultralytics's YOLOv8. There are pretty good examples on its [Github Readme](https://github.com/Deci-AI/super-gradients) and [here](https://www.kaggle.com/general/406701).

## Installation

```bash
$ pip install -r requirements.txt
```

If you would like to skip the training part, download the `best.pt` from [HERE](https://drive.google.com/drive/folders/1aZ4MpL7zXARpdr7hky6iucenSEoaDH9W?usp=sharing) and place it into the `model` folder.

## Training

1. Download [BUP20](http://agrobotics.uni-bonn.de/sweet_pepper_dataset/) dataset and place `BUP20.tar.gz` into the folder `train/1_place_bup20_tar_gz_here`

2. Run the notebook `train/2_bup20_to_yolov8.ipynb ` to create yolov8 dataset.
3. Run the script `train/3_train.py` for training. This should take about 15-20 mins.
4. Copy the model into

## Prediction

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

  ```
  $ python predict_images.py
  ```

  
