#!/usr/bin/env python3
import roslib
roslib.load_manifest('yolo_segmentation_ros')

import sys
import os
import rospy
import rospkg
import random

import cv2
from ultralytics import YOLO
import torch
from collections import deque
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage, RegionOfInterest

FOUND_yolact_ros_msgs = True
try:
  from yolact_ros_msgs.msg import Detections, Detection, Box, Mask
except:
  FOUND_yolact_ros_msgs = False
  print("yolact_ros_msgs not found!")

FOUND_mask_rcnn_ros = True
try:
  from mask_rcnn_ros.msg import Result
except:
  FOUND_mask_rcnn_ros = False
  print("mask_rcnn_ros not found!")

FOUND_panoptic_mapping_msgs = True
try:
  from panoptic_mapping_msgs.msg import DetectronLabel, DetectronLabels
except:
  FOUND_panoptic_mapping_msgs = False
  print("panoptic_mapping_msgs not found!")

class FakePublisher:
  def __init__(self):
    pass
  def get_num_connections(self):
    return 0



# GLASBEY color table [256][3]
GLASBEY = [(255,255,255), (0,0,255), (255,0,0), (0,255,0), (0,0,51), (255,0,182), (0,83,0), (255,211,0), (0,159,255), (154,77,66), (0,255,190), (120,63,193), (31,150,152), (255,172,253), (177,204,113), (241,8,92), (254,143,66), (221,0,255), (32,26,1), (114,0,85), (118,108,149), (2,173,36), (200,255,0), (136,108,0), (255,183,159), (133,133,103), (161,3,0), (20,249,255), (0,71,158), (220,94,147), (147,212,255), (0,76,255), (0,66,80), (57,167,106), (238,112,254), (0,0,100), (171,245,204), (161,146,255), (164,255,115), (255,206,113), (71,0,21), (212,173,197), (251,118,111), (171,188,0), (117,0,215), (166,0,154), (0,115,254), (165,93,174), (98,132,2), (0,121,168), (0,255,131), (86,53,0), (159,0,63), (66,45,66), (255,242,187), (0,93,67), (252,255,124), (159,191,186), (167,84,19), (74,39,108), (0,16,166), (145,78,109), (207,149,0), (195,187,255), (253,68,64), (66,78,32), (106,1,0), (181,131,84), (132,233,147), (96,217,0), (255,111,211), (102,75,63), (254,100,0), (228,3,127), (17,199,174), (210,129,139), (91,118,124), (32,59,106), (180,84,255), (226,8,210), (0,1,20), (93,132,68), (166,250,255), (97,123,201), (98,0,122), (126,190,58), (0,60,183), (255,253,0), (7,197,226), (180,167,57), (148,186,138), (204,187,160), (55,0,49), (0,40,1), (150,122,129), (39,136,38), (206,130,180), (150,164,196), (180,32,128), (110,86,180), (147,0,185), (199,48,61), (115,102,255), (15,187,253), (172,164,100), (182,117,250), (216,220,254), (87,141,113), (216,85,34), (0,196,103), (243,165,105), (216,255,182), (1,24,219), (52,66,54), (255,154,0), (87,95,1), (198,241,79), (255,95,133), (123,172,240), (120,100,49), (162,133,204), (105,255,220), (198,82,100), (121,26,64), (0,238,70), (231,207,69), (217,128,233), (255,211,209), (209,255,141), (36,0,3), (87,163,193), (211,231,201), (203,111,79), (62,24,0), (0,117,223), (112,176,88), (209,24,0), (0,30,107), (105,200,197), (255,203,255), (233,194,137), (191,129,46), (69,42,145), (171,76,194), (14,117,61), (0,30,25), (118,73,127), (255,169,200), (94,55,217), (238,230,138), (159,54,33), (80,0,148), (189,144,128), (0,109,126), (88,223,96), (71,80,103), (1,93,159), (99,48,60), (2,206,148), (139,83,37), (171,0,255), (141,42,135), (85,83,148), (150,255,0), (0,152,123), (255,138,203), (222,69,200), (107,109,230), (30,0,68), (173,76,138), (255,134,161), (0,35,60), (138,205,0), (111,202,157), (225,75,253), (255,176,77), (229,232,57), (114,16,255), (111,82,101), (134,137,48), (99,38,80), (105,38,32), (200,110,0), (209,164,255), (198,210,86), (79,103,77), (174,165,166), (170,45,101), (199,81,175), (255,89,172), (146,102,78), (102,134,184), (111,152,255), (92,255,159), (172,137,178), (210,34,98), (199,207,147), (255,185,30), (250,148,141), (49,34,78), (254,81,97), (254,141,100), (68,54,23), (201,162,84), (199,232,240), (68,152,0), (147,172,58), (22,75,28), (8,84,121), (116,45,0), (104,60,255), (64,41,38), (164,113,215), (207,0,155), (118,1,35), (83,0,88), (0,82,232), (43,92,87), (160,217,146), (176,26,229), (29,3,36), (122,58,159), (214,209,207), (160,100,105), (106,157,160), (153,219,113), (192,56,207), (125,255,89), (149,0,34), (213,162,223), (22,131,204), (166,249,69), (109,105,97), (86,188,78), (255,109,81), (255,3,248), (255,0,73), (202,0,35), (67,109,18), (234,170,173), (191,165,0), (38,44,51), (85,185,2), (121,182,158), (254,236,212), (139,165,89), (141,254,193), (0,60,43), (63,17,40), (255,221,246), (17,26,146), (154,66,84), (149,157,238), (126,130,72), (58,6,101), (189,117,101)]
CLASS_NAMES = ['pepper'] * 80 # you only pepper once!

#CLASS_NAMES = ['person', 'pepper', 'car', 'peduncle', 'airplane', 'bus', 'train'] * 80 # you only pepper once!

class YoloNode:
  def __init__(self, p):
    # get params
    self.image_topic          = p["image_topic"]
    self.model_path           = p["model_path"]
    self.device               = torch.device(p["device"])
    self.fps_limit            = p["fps_limit"]
    self.use_compressed_image = p["use_compressed_image"]
    self.score_threshold      = p["score_threshold"]
    self.retina_masks         = p["retina_masks"]
    self.iou                  = p["iou"]
    self.tracking             = p["tracking"]
    self.use_our_visualizer   = p["use_our_visualizer"]
    self.display_bboxes       = p["display_bboxes"]
    self.display_text         = p["display_text"]
    self.display_masks        = p["display_masks"]
    self.display_fps          = p["display_fps"]
    self.display_scores       = p["display_scores"]
    self.verbose              = p["verbose"]

    # process params
    if self.fps_limit > 0:
      self.rate = rospy.Rate(self.fps_limit)
    else:
      self.rate = None

    self.model = YOLO(self.model_path)
    for k,v in self.model.names.items():
      self.model.names[k] = "pepper"

    # Other things
    self.processing_times_queue = deque(maxlen=10)
    self.instance_counter = 1 # TODO: handle overflow? not quite possible.

    # Publishers
    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("~visualization", Image, queue_size=1)

    if FOUND_mask_rcnn_ros:
      self.result_pub = rospy.Publisher('/mask_rcnn/result', Result, queue_size=1)
    else:
      self.result_pub = FakePublisher()

    if FOUND_yolact_ros_msgs:
      self.detections_pub = rospy.Publisher("~detections", Detections, queue_size=1)
    else:
      self.detections_pub = FakePublisher()

    if FOUND_panoptic_mapping_msgs:
      self.panoptic_image_pub = rospy.Publisher("/camera/segmentation_image", Image, queue_size=1)
      self.panoptic_labels_pub = rospy.Publisher("/camera/segmentation_labels", DetectronLabels, queue_size=1)
    else:
      self.panoptic_image_pub  = FakePublisher()
      self.panoptic_labels_pub = FakePublisher()

    # Subscribers
    if self.use_compressed_image is None:
      self.use_compressed_image = self.image_topic.endswith('/compressed')
    if self.use_compressed_image:
      self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.callback_compressed, queue_size=2, buff_size=2**24)
    else:
      self.image_sub = rospy.Subscriber(self.image_topic, Image, self.callback, queue_size=2, buff_size=2**24)

#############################

  def process_image(self, cv_image, header):
    #cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE) # TODO
    height, width, channels = cv_image.shape
    # Run inference
    if self.tracking:
      result = self.model.track(cv_image, retina_masks=self.retina_masks, conf=self.score_threshold, iou=self.iou, persist=True, device=self.device, verbose=self.verbose)[0]
    else:
      result = self.model(cv_image, retina_masks=self.retina_masks, conf=self.score_threshold, iou=self.iou, device=self.device, verbose=self.verbose)[0]

    # Extract result
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    classes = np.ones_like(classes) # a trick to override class id's to 1
    scores = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = None
    if self.tracking and result.boxes.id is not None:
      ids = result.boxes.id.detach().cpu().numpy().astype(int) + 1 # +1 to make sure to start id's from one instead of zero
    masks = []
    if result.masks is not None:
      masks = result.masks.data.detach().cpu().numpy() > 0.5

    if not FOUND_mask_rcnn_ros and not FOUND_yolact_ros_msgs and not FOUND_panoptic_mapping_msgs:
      rospy.logerr("mask_rcnn_ros, yolact_ros_msgs and panoptic_mapping_msgs not found! Can't publish detections.")

    # Publish "Detections" msg
    if self.detections_pub.get_num_connections() > 0:
      dets = self._build_detections_msg(classes, scores, boxes, masks, ids, header)
      self.detections_pub.publish(dets)

    # Publish "Result" msg
    if self.result_pub.get_num_connections() > 0:
      result_msg = self._build_result_msg(classes, scores, boxes, masks, ids, header)
      self.result_pub.publish(result_msg)

    # Publish "DetectronLabels msg with the segmentation image"
    if self.panoptic_image_pub.get_num_connections()> 0 or self.panoptic_labels_pub.get_num_connections() > 0:
      seg_image_msg, det_label_msg = self. _build_detectron_label_msg(classes, scores, boxes, masks, ids, header, height, width)
      self.panoptic_image_pub.publish(seg_image_msg)
      self.panoptic_labels_pub.publish(det_label_msg)

    # Publish visualization
    if self.image_pub.get_num_connections() > 0:
      try:
        if self.use_our_visualizer:
          vis_img = self.generate_visualization(cv_image, classes, scores, boxes, masks, ids)
        else:
          vis_img = result.plot()
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
      except CvBridgeError as e:
        print(e)

    self.instance_counter += len(classes)

    if self.rate is not None:
      self.rate.sleep()

#############################

  def callback(self, data):
    if self.image_pub.get_num_connections() == 0 and self.result_pub.get_num_connections() == 0 and self.detections_pub.get_num_connections() == 0 and self.panoptic_image_pub.get_num_connections() == 0 and self.panoptic_labels_pub.get_num_connections() == 0:
        return
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    self.process_image(cv_image, data.header)

  def callback_compressed(self, data):
    if self.image_pub.get_num_connections() == 0 and self.result_pub.get_num_connections() == 0 and self.detections_pub.get_num_connections() == 0 and self.panoptic_image_pub.get_num_connections() == 0 and self.panoptic_labels_pub.get_num_connections() == 0:
        return
    np_arr = np.fromstring(data.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    self.process_image(cv_image, data.header)

#############################

  def generate_visualization(self, image, classes, scores, boxes, masks, ids=None):
      res_image = image
      instance_counter = self.instance_counter # self.instance_counter is modified by process_image
      for detnum in range(len(classes)):
        if ids is None:
          #color = GLASBEY[instance_counter % 256] # psychedelic peppers
          color = GLASBEY[instance_counter - self.instance_counter] # unique colors in the frame
          #color = GLASBEY[42] # single color in the frame
          instance_counter += 1
        else:
          color = GLASBEY[ids[detnum] % 256]

        x1, y1, x2, y2 = boxes[detnum].astype(int)
        if self.display_bboxes:
           res_image = cv2.rectangle(res_image, (x1, y1), (x2, y2), color, 1)

        if self.display_text:
           label = CLASS_NAMES[classes[detnum]]
           text_str = '%s: %.2f' % (label, scores[detnum]) if self.display_scores else label
           res_image = cv2.putText(res_image, text_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        if self.display_masks:
          res_image = self.apply_mask(res_image, masks[detnum,:,:], color)

      if self.display_fps:
        self.processing_times_queue.append(rospy.get_rostime())
        if len(self.processing_times_queue) > 1:
            self.fps = (len(self.processing_times_queue) - 1) / (self.processing_times_queue[-1] - self.processing_times_queue[0]).to_sec()
            res_image = cv2.putText(res_image, '%.1f FPS' % self.fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

      return res_image

  def apply_mask(self, image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, (1 - alpha) * image[:, :, c] + alpha * mask * color[c], image[:, :, c])
    return image

  def _build_detections_msg(self, classes, scores, boxes, masks, ids, image_header):
    dets_msg = Detections()
    instance_counter = self.instance_counter # self.instance_counter is modified by process_image
    for detnum in range(len(classes)):

      ###################
      # TODO: When to Detection message is modified to store instance ids, use instance_id for publishing object id
      if ids is None:
        instance_id = instance_counter
        instance_counter += 1
      else:
        instance_id = ids[detnum]
      ###################

      det = Detection()
      det.class_name = CLASS_NAMES[classes[detnum]]
      det.score = float(scores[detnum])
      x1, y1, x2, y2 = boxes[detnum].astype(int)
      det.box.x1 = int(x1)
      det.box.y1 = int(y1)
      det.box.x2 = int(x2)
      det.box.y2 = int(y2)
      mask = masks[detnum,y1:y2,x1:x2]
      det.mask.mask = np.packbits(mask.astype(bool)).tolist()
      det.mask.height = int(y2 - y1)
      det.mask.width = int(x2 - x1)
      dets_msg.detections.append(det)

    dets_msg.header = image_header
    return dets_msg

  def _build_result_msg(self, classes, scores, boxes, masks, ids, image_header):
    #print("Building result_msg")
    result_msg = Result()
    result_msg.header = image_header
    instance_counter = self.instance_counter # self.instance_counter is modified by process_image
    for i in range(len(classes)):
        class_id = classes[i]
        if ids is None:
          instance_id = instance_counter
          instance_counter += 1
        else:
          instance_id = ids[i]
        if CLASS_NAMES[class_id] == "pepper":
          box = RegionOfInterest()
          x1, y1, x2, y2 = boxes[i].astype(int)

          box.x_offset = int(x1)
          box.y_offset = int(y1)
          box.height = int(y2) - int(y1)
          box.width = int(x2) - int(x1)

          # TODO: There is a workaround for signed/unsigned integer problem down below. Actually, bboxes are different than region of interests, because;
          #         - BBox defines an area on the spatial domain. There are no limits.
          #         - Region of interest defines an area limited by the image.
          # So... Should we limit bbox values to between [0,width] and [0,height] values?
          # Workaround for "rospy.exceptions.ROSSerializationException: field boxes[].y_offset must be unsigned integer type"
          if box.x_offset < 0:
            box.width -= box.x_offset
            box.x_offset = 0
          if box.y_offset < 0:
            box.height -= box.y_offset
            box.y_offset = 0

          result_msg.boxes.append(box)
          result_msg.class_ids.append(class_id)

          # for backward compatibility
          if hasattr(result_msg, "instance_ids"):
            result_msg.instance_ids.append(instance_id)

          class_name = CLASS_NAMES[class_id]
          result_msg.class_names.append(class_name)

          score = float(scores[i])
          result_msg.scores.append(score)

          mask_op = masks[i]
          mask = Image()
          mask.header = image_header
          
          mask.height = mask_op.shape[0]
          mask.width = mask_op.shape[1]
          mask.encoding = "mono8"
          mask.is_bigendian = False
          mask.step = mask.width
          mask.data = mask_op.tobytes()
          result_msg.masks.append(mask)

    return result_msg

  def _build_detectron_label_msg(self, classes, scores, boxes, masks, ids, image_header, height, width):
    if(len(masks) > 0):
      xor_image = np.zeros(masks[0].shape, np.uint8)
      zero_image = np.zeros(xor_image.shape, np.uint8)
    else:
      shape = [height, width]
      xor_image = np.zeros(shape, np.uint8)
      zero_image = np.zeros(xor_image.shape, np.uint8)
    instance_counter = self.instance_counter
    det_labels = DetectronLabels()
    det_labels.header = image_header

    #print(masks[0].shape)
    bg_label = DetectronLabel()
    bg_label.id = 0
    bg_label.is_thing = False
    bg_label.score = 0.9
    bg_label.category_id = 0
    bg_label.instance_id = 0
    det_labels.labels.append(bg_label)
    if(len(masks) > 0):
      for i in range(len(classes)):
        class_id = classes[i]

        if ids is None:
          instance_id = instance_counter
          instance_counter += 1
        else:
          instance_id = ids[i]

        if CLASS_NAMES[class_id] == "pepper":
            label = DetectronLabel()
            label.id = instance_id #i+1
            label.is_thing = True
            label.score = float(scores[i])
            label.category_id = class_id
            label.instance_id = instance_id
            det_labels.labels.append(label)

            masks[i] = label.id * masks[i]
            xor_image = 1*np.logical_xor(xor_image, masks[i])

      for i in range(len(classes)):
        class_id = classes[i]
        if CLASS_NAMES[class_id] == "pepper":
           zero_image = zero_image + np.multiply(xor_image, masks[i])

    seg_image = Image()
    seg_image.header = image_header
    seg_image.height = zero_image.shape[0]
    seg_image.width =  zero_image.shape[1]
    seg_image.encoding = "mono8"
    seg_image.is_bigendian = False
    seg_image.step = seg_image.width
    temp = zero_image.astype(np.uint8)
    seg_image.data = temp.tobytes()

    return seg_image, det_labels

def main(args):
  rospy.init_node('yolo_segmentation_ros')
  rospack = rospkg.RosPack()
  agrobot_path = rospack.get_path('yolo_segmentation_ros')
  model_path = agrobot_path + "/model/best.pt"

  # Get parameters
  p = {}
  p["image_topic"]        = "input_image"
  p["model_path"]         = rospy.get_param('~model_path', model_path)
  p["device"]             = rospy.get_param('~device', "cuda:0")
  p["fps_limit"]          = rospy.get_param('~fps_limit', -1.0)
  if rospy.has_param("~use_compressed_image"):
    p["use_compressed_image"] = rospy.get_param('~use_compressed_image')
  else:
    p["use_compressed_image"] = None
  p["score_threshold"]    = rospy.get_param('~score_threshold', 0.75)
  p["retina_masks"]       = rospy.get_param('~retina_masks', True)
  p["iou"]                = rospy.get_param('~iou', 0.5) #  intersection over union (IoU) threshold for NMS
  p["tracking"]           = rospy.get_param('~tracking', True)
  p["use_our_visualizer"] = rospy.get_param('~use_our_visualizer', True)
  p["display_bboxes"]     = rospy.get_param('~display_bboxes', True)
  p["display_text"]       = rospy.get_param('~display_text', True)
  p["display_masks"]      = rospy.get_param('~display_masks', True)
  p["display_fps"]        = rospy.get_param('~display_fps', True)
  p["display_scores"]     = rospy.get_param('~display_scores', True)
  p["verbose"]            = rospy.get_param('~verbose', True)

  # Run the node!
  yn = YoloNode(p)
  rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
