# yolo_segmentation_ros

Alternative to `agrobot_mrcnn_ros` and `mask_rcnn_ros`. Uses `mask_rcnn_ros/Result` for publishing the results.

## Installation

```bash
$ pip install -r requirements.txt
```

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

- Predict using your webcam:

  ```bash
  $ python predict_webcam.py
  ```

- Predict using a video:

  ```bash
  $ python predict_video.py.py
  ```

  
