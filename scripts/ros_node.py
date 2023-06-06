#!/usr/bin/env python3
import roslib
roslib.load_manifest('yolo_segmentation_ros')

import cv2
from ultralytics import YOLO
import torch

import sys
import os
import rospy
import rospkg

from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros_msgs.msg import Detections
from yolact_ros_msgs.msg import Detection
from yolact_ros_msgs.msg import Box
from yolact_ros_msgs.msg import Mask

from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from mask_rcnn_ros.msg import Result

CLASS_NAMES = ['background', 'pepper']

class YoloNode:
  def __init__(self, image_topic, model_path, device="cuda", fps_limit=-1, use_compressed_image=None):
    self.device = torch.device(device)
    self.model = YOLO(model_path) # TODO: device=self.device
    if fps_limit > 0:
      self.rate = rospy.Rate(fps_limit)
    else:
      self.rate = None

    # a dirty hack to replace "pepper" with "orange"
    for k,v in self.model.names.items():
      if v == "orange":
        self.model.names[k] = "pepper"
        break

    self.bridge = CvBridge()

    self.image_pub = rospy.Publisher("~visualization", Image, queue_size=1)
    self.result_pub = rospy.Publisher('/mask_rcnn/result', Result, queue_size=1)
    self.detections_pub = rospy.Publisher("~detections", Detections, queue_size=1)

    if use_compressed_image is None:
      self.use_compressed_image = image_topic.endswith('/compressed')
    if self.use_compressed_image:
      self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.callback_compressed, queue_size=1, buff_size=2**24)
    else:
      self.image_sub = rospy.Subscriber(image_topic, Image, self.callback, queue_size=1, buff_size=2**24)

#############################

  def process_image(self, cv_image):
    # Run YOLOv8 inference
    cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
    print(cv_image.shape)
    results = self.model.track(cv_image, retina_masks=True, conf=0.75, iou=0.5, persist=True, device=self.device)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    if self.image_pub.get_num_connections() > 0:
      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
      except CvBridgeError as e:
        print(e)

    if self.rate is not None:
      self.rate.sleep()

#############################

  def callback(self, data):
    if self.image_pub.get_num_connections() == 0 and self.result_pub.get_num_connections() == 0 and self.detections_pub.get_num_connections() == 0:
        return

    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    self.process_image(cv_image)

  def callback_compressed(self, data):
    if self.image_pub.get_num_connections() == 0 and self.result_pub.get_num_connections() == 0 and self.detections_pub.get_num_connections() == 0:
        return

    np_arr = np.fromstring(data.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    self.process_image(cv_image)

#############################

  def create_hsv_mask(self, cv_image):
    image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 30, 30])
    upper_red = np.array([25, 255, 255])

    image_red = cv2.inRange(image_hsv, lower_red, upper_red)
    kernel = np.ones((3,3),np.uint8)
    #image_red = cv2.dilate(image_red,kernel,iterations = 1)
    #image_red = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel, iterations=1)
    image_red = cv2.erode(image_red, kernel,iterations = 2)
    contours, _ = cv2.findContours(image_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image_red.shape, np.uint8)
    i = 0
    masks =[]
    bboxes = []
    scores = []
    classes = []
    class_names = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
      if cv2.contourArea(cnt) > 10:
        #print(cv2.contourArea(cnt))
        i = i+1
        color1 = list(np.random.choice(range(256), size=3))
        color =[int(color1[0]), int(color1[1]), int(color1[2])]  
        rect = cv2.boundingRect(cnt)
        x,y,w,h = rect
        rect_box = np.array([x, y, x+w, y+h])
        # print(rect)
        # print("x", x)
        # print("y", y)
        # print("w", w)
        # print("h", h)
        cv2.drawContours(mask, [cnt], 0, (255,255,255),-1)
        #cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),2)
        #cv2.namedWindow("mask_image")
        #cv2.imshow('mask_image', mask) 
        #cv2.waitKey(0)
        mask_arr = np.asarray(mask)
        mask_bool = mask_arr.astype(np.bool) 
        #print(np.transpose(np.nonzero(mask_bool == True)))
        masks.append(np.array(mask_bool))
        bboxes.append(rect_box)
        scores.append(0.9)
        classes.append(1)
        class_names.append("pepper")
      # if i == 1:
      #   cv2.drawContours(mask, [cnt], 0, (255,255,255),-1)
      #   cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),2)
      #   cv2.namedWindow("mask_image")
      #   cv2.imshow('mask_image', mask) 
      #   cv2.waitKey(0)
    return classes, scores, bboxes, masks
      # if i == 1:
      #   cv2.drawContours(mask, [cnt], 0, (255,255,255),-1)
      #   cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),2)
      #   cv2.namedWindow("mask_image")
      #   cv2.imshow('mask_image', mask) 
      #   cv2.waitKey(0)
    return classes, scores, bboxes, masks

  def apply_mask(self, image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, (1 - alpha) * image[:, :, c] + alpha * mask * color[c], image[:, :, c])
    return image

  def apply_cropped_mask(self, image, box, mask, color, alpha=0.5):
    x1, y1, x2, y2 = box
    for c in range(3):
        image[y1:y2, x1:x2, c] = np.where(mask == 1, (1 - alpha) * image[y1:y2, x1:x2, c] + alpha * mask * color[c], image[y1:y2, x1:x2, c])
    return image

  def generate_detections_msg(self, classes, scores, boxes, masks, image_header, sub_classes, sub_scores):
    dets_msg = Detections()
    for detnum in range(len(classes)):
      det = Detection()
      det.class_name = self.labels[classes[detnum]]
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

      #encoding_correct = True
      #for x in range(det.mask.width):
      #    for y in range(det.mask.height):
      #        if bool(masks[detnum,y1+y,x1+x]) != mask_utils.test(det.mask, x, y):
      #            encoding_correct = False
      #print('Encoding correct: ' + str(encoding_correct))

    dets_msg.header = image_header
    return dets_msg

  def _build_result_msg(self, classes, scores, boxes, masks, image_header, sub_classes, sub_scores):
    #print("Building result_msg")
    result_msg = Result()
    result_msg.header = image_header
    for i in range(len(classes)):
        class_id = classes[i]
        if self._class_names[class_id] == "pepper":
          box = RegionOfInterest()
          x1, y1, x2, y2 = boxes[i].astype(int)
          box.x_offset = int(x1)
          box.y_offset = int(y1)
          box.height = int(y2) - int(y1)
          box.width = int(x2) - int(x1)
          result_msg.boxes.append(box)

         
          result_msg.class_ids.append(class_id)

          class_name = self._class_names[class_id]
          result_msg.class_names.append(class_name)

          score = float(scores[i])
          result_msg.scores.append(score)

          mask_op = masks[i]
          #print(type(mask_op))
          #print(mask_op.shape)
          #print(np.transpose(np.nonzero(mask_op == True)))
          mask = Image()
          mask.header = image_header
          
          mask.height = mask_op.shape[0]
          mask.width = mask_op.shape[1]
          mask.encoding = "mono8"
          mask.is_bigendian = False
          mask.step = mask.width
          mask.data = mask_op.tobytes()
          result_msg.masks.append(mask)
    
    #print("Returning result_msg")
    return result_msg

  def generate_visualization(self, image, classes, scores, boxes, masks, sub_classes, sub_scores, simulation):
      res_image = image
      for detnum in range(len(classes)):
        # Visualization
        color = self.glasbey[classes[detnum] % 256]
        #print("type of box: ", type(boxes[detnum]))
        #print(boxes[detnum])
        x1, y1, x2, y2 = boxes[detnum].astype(int)
        if self.display_bboxes:
           res_image = cv2.rectangle(res_image, (x1, y1), (x2, y2), color, 1)
        if self.display_text:
           label = self.labels[classes[detnum]]
           text_str = '%s: %.2f' % (label, scores[detnum]) if self.display_scores else label
           res_image = cv2.putText(res_image, text_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
           if sub_classes is not None:
               sub_label = self.sub_labels[sub_classes[detnum]]
               sub_score = sub_scores[detnum, sub_classes[detnum]]
               sub_text_str = '%s: %.2f' % (sub_label, sub_score) if self.display_scores else sub_label
               res_image = cv2.putText(res_image, sub_text_str, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        if self.display_masks:
          if simulation == False:
            res_image = self.apply_mask(res_image, masks[detnum,:,:], color)
          else:
            res_image = self.apply_mask(res_image, masks[detnum], color)

      if self.display_fps:
        self.processing_times_queue.append(rospy.get_rostime())
        if len(self.processing_times_queue) > 1:
            self.fps = (len(self.processing_times_queue) - 1) / (self.processing_times_queue[-1] - self.processing_times_queue[0]).to_sec()
            res_image = cv2.putText(res_image, '%.1f FPS' % self.fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

      return res_image

  def evalimage(self, cv_image, image_header, rotate_image_for_inference, simulation):
    res_image = None
    classes, scores, boxes, masks = None, None, None, None
    sub_classes, sub_scores = None, None

    if simulation == False:
      with torch.no_grad():
        frame = torch.from_numpy(cv_image).to(self.device).permute(2, 0, 1) / 255.0

        # Apply rotations to the image
        if rotate_image_for_inference != 0:
          frame = torch.rot90(frame, k=rotate_image_for_inference//90, dims=[2,1])

        # Inference
        output = self.model([frame])[0]

        # (PART-1/2) Apply inverse rotations to the outputs
        if len(output['scores']) > 0:
          if rotate_image_for_inference != 0:
            output['masks'] = torch.rot90(output['masks'], k=-rotate_image_for_inference//90, dims=[3,2])

        # Move from GPU to CPU memory
        classes, scores, boxes, masks = output['labels'].cpu().numpy(), output['scores'].cpu().numpy(), output['boxes'].cpu().numpy(), output['masks'].cpu().numpy()

        # (PART-2/2) Apply inverse rotations to the outputs
        if len(scores) > 0:
          cv_rows = cv_image.shape[0]
          cv_cols = cv_image.shape[1]
          x1 = boxes[:, 0]
          y1 = boxes[:, 1]
          x2 = boxes[:, 2]
          y2 = boxes[:, 3]
          if rotate_image_for_inference == 0:
            pass
          elif rotate_image_for_inference == 90:
            boxes[:, :] = np.array([y1, cv_rows-x2, y2, cv_rows-x1], dtype=boxes.dtype).T
          elif rotate_image_for_inference == 180:
            boxes[:, :] = np.array([cv_cols-x2, cv_rows-y2, cv_cols-x1, cv_rows-y1], dtype=boxes.dtype).T
          elif rotate_image_for_inference == 270:
            boxes[:, :] = np.array([cv_cols-y2, x1, cv_cols-y1, x2], dtype=boxes.dtype).T

        if self.use_subclasses:
            sub_classes, sub_scores = output['sub_labels'].cpu().numpy(), output['sub_scores'].cpu().numpy()

      # compute score cutoff index
      ci = np.searchsorted(scores, self.score_threshold, sorter=np.arange(len(scores)-1, -1, -1))
      ci = len(scores) - ci
      if self.top_k > 0:
        ci = min(ci, self.top_k)

      classes = classes[:ci]

      # compute binary masks
      masks = masks[:ci, 0, :, :] >= 0.5 # TODO
      # print("mask type: ", type(masks[0]))
      # print("shape: ", masks[0].shape)
      # #print(masks[0])
      # print("box: ", boxes[0])
    else:
      classes, scores, boxes, masks = self.create_hsv_mask(cv_image)
      # print("mask type: ", type(masks[0]))
      # print("shape: ", masks[0].shape)
      # print("boxes: ", boxes)
      #print(masks[0])
    #self.display_visualization = True
    if self.image_pub.get_num_connections() > 0 or self.display_visualization:
      res_image = self.generate_visualization(cv_image, classes, scores, boxes, masks, sub_classes, sub_scores, simulation)

    if self.detections_pub.get_num_connections() > 0:
      dets = self.generate_detections_msg(classes, scores, boxes, masks, image_header, sub_classes, sub_scores)
      self.detections_pub.publish(dets)

    result_msg = self._build_result_msg(classes, scores, boxes, masks, image_header, sub_classes, sub_scores)
    self.result_pub.publish(result_msg)

    if self.image_pub.get_num_connections() > 0:
      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(res_image, "bgr8"))
      except CvBridgeError as e:
        print(e)


def main(args):
  rospy.init_node('yolo_segmentation_ros')
  rospack = rospkg.RosPack()
  agrobot_path = rospack.get_path('yolo_segmentation_ros')
  model_path = agrobot_path + "/model/best.pt"

  image_topic = "/trollomatic/camera1/color/image_raw"

  # Parameters
  model_path = rospy.get_param('~model_path', model_path)
  device = rospy.get_param('~device', "cuda:0")
  fps_limit = rospy.get_param('~fps_limit', -1.0)

  yn = YoloNode(image_topic, model_path, device, fps_limit)
  rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
