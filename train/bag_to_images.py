#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    # parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    # parser.add_argument("bag_file", help="Input ROS bag.")
    # parser.add_argument("output_dir", help="Output directory.")
    # parser.add_argument("image_topic", help="Image topic.")

    #args = parser.parse_args()

    #print ()"Extract images from %s on topic %s into %s" % (args.bag_file,
    #                                                      args.image_topic, args.output_dir))
    bag_file = "/media/rohit/data/armbags/20221013/try4_row5_no_map/armbag_2022-10-13-14-21-51.bag"
    output_dir = "/home/rohit/Documents/armbag_dataset/"
    image_topic = "/camera/color/image_raw"
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = -1
    count2 = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        count += 1
        if count % 50 == 0:
            cropped_image = cv_img[60:420, 0:640]

            cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count2), cropped_image)
            print("Wrote image ",count2)
            count2 = count2 + 1


    bag.close()

    return

if __name__ == '__main__':
    main()