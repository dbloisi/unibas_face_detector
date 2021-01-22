#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('unibas_distance_from_camera')
import sys
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class get_face_distance_from_camera:

  def __init__(self):    
  
    
  	
    self.mm = 1000
    
     
    self.bridge = CvBridge()
    
    self.camera_info_sub = message_filters.Subscriber('/camera/rgb/camera_info', CameraInfo)
    
        	
    self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw",Image)
    self.depth_sub = message_filters.Subscriber("/camera/depth_registered/image_raw",Image)
        
    self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info_sub], queue_size=10, slop=0.5)
    self.ts.registerCallback(self.callback)
        
    self.pub = rospy.Publisher('/unibas_face_detector/faces', Image, queue_size=1)	

  def callback(self, rgb_data, depth_data, camera_info):
    
    try:
      camera_info_K = np.array(camera_info.K)
    
      m_fx = camera_info.K[0];
      m_fy = camera_info.K[4];
      m_cx = camera_info.K[2];
      m_cy = camera_info.K[5];
      inv_fx = 1. / m_fx;
      inv_fy = 1. / m_fy;
    
    
      cv_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
      depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
      depth_array = np.array(depth_image, dtype=np.float32)
      cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
      depth_8 = (depth_array * 255).round().astype(np.uint8)
      cv_depth = np.zeros_like(cv_rgb)
      cv_depth[:,:,0] = depth_8
      cv_depth[:,:,1] = depth_8
      cv_depth[:,:,2] = depth_8
      
      face_cascade = cv2.CascadeClassifier('/home/bloisi/catkin_ws/src/unibas_distance_from_camera/haarcascade/haarcascade_frontalface_default.xml')
      gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      rgb_height, rgb_width, rgb_channels = cv_rgb.shape
      for (x,y,w,h) in faces:
        cv2.rectangle(cv_rgb,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(cv_depth,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(cv_rgb,(x+30,y+30),(x+w-30,y+h-30),(0,0,255),2)
        cv2.rectangle(cv_depth,(x+30,y+30),(x+w-30,y+h-30),(0,0,255),2)
        roi_depth = depth_array[y+30:y+h-30, x+30:x+w-30]
        
        n = 0
        sum = 0
        for i in range(0,roi_depth.shape[0]):
            for j in range(0,roi_depth.shape[1]):
                value = roi_depth.item(i, j)
                if value > 0.:
                    n = n + 1
                    sum = sum + value
        
        d = sum / n
        
        point_z = d * self.mm;
        point_x = ((x + w/2) - m_cx) * point_z * inv_fx;
        point_y = ((y + h/2) - m_cy) * point_z * inv_fy;
        
        print(str(point_x) + " " + str(point_y) + " " + str(point_z))
            
    except CvBridgeError as e:
      print(e)
      
    rgbd = np.concatenate((cv_rgb, cv_depth), axis=1)

    #convert opencv format back to ros format and publish result
    try:
      faces_message = self.bridge.cv2_to_imgmsg(rgbd, "bgr8")
      self.pub.publish(faces_message)
    except CvBridgeError as e:
      print(e)
    

def main(args):
  rospy.init_node('unibas_distance_from_camera', anonymous=True)
  fd = get_face_distance_from_camera()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)

