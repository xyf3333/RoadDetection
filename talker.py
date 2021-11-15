#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 09:52:57 2021

@author: py
"""
import os
import sys
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import math
from sensor_msgs.msg import LaserScan


try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    sys.exit(1)

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Set model here:
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'/home/py/catkin_ws/src/transferlearning_course/transferlearning_course/models/research/object_detection/course_model_1')
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'

#Set the label map file here:
LABEL_NAME = 'course.pbtxt'
PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]),'/home/py/catkin_ws/src/transferlearning_course/transferlearning_course/models/research/object_detection/data', LABEL_NAME)

#Set the number of classes here
NUM_CLASSES = 2

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5

'''
#Used when testing on saved images: 
PATH_TO_TEST_IMAGES_DIR = 'object_detection'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(0, 501) ]
'''

class Detector:
    def __init__(self):
        #tensorflow session is initiated
        self.i = 0
        self.sess = tf.Session(graph=detection_graph,config=config)
        #self.throttle = 0

    def pred(self, image):
        height, width, depth = image.shape
        area = height*width
        aplha = 0.4
       
        #Low lightning conditions:
        overlay = image.copy()#rgb 
        output = image#rgb
        cv2.rectangle(overlay, (0, 0), (width, height),(246, 245, 255), -1)
        cv2.addWeighted(overlay, aplha, output, 1 - aplha,0, output)

        output=cv2.cvtColor(output, cv2.COLOR_BGR2RGB)#BGR
        cam_np = np.asarray(output)#has to be rgb
        cam_np_expanded = np.expand_dims(cam_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: cam_np_expanded})
        B = np.squeeze(boxes)
        S = np.squeeze(scores)
        C = np. squeeze(classes)
        #print ("Classes:",C[0])
        #print ("Scores:",S)
        if scores[0][0] >= 0.5:
            cam_box1 = boxes[0][0].tolist()
        else:
            cam_box1 = [0,0,0,0]



        cam_ymin1, cam_xmin1, cam_ymax1, cam_xmax1 = cam_box1
        cam_x_center_box1 = int(cam_xmin1 * width + (cam_xmax1 - cam_xmin1) * width * 0.5)
        cam_y_center_box1 = int(cam_ymin1 * height + (cam_ymax1 - cam_ymin1) * height * 0.5)
        size1 = abs(cam_xmax1-cam_xmin1) * abs(cam_ymax1-cam_ymin1)
        size1 = 100*size1/area

        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            output,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
        
        img=cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  
        cv2.resize("sign_detection", (640, 480))
        cv2.imshow("sign_detection", img)

        cv2.waitKey(3)

        return C, S, B





def check_zero_in_matrix(arr):
    """
    Check if all 2D numpy array contains only 0
    :param arr: a 2D array
    :return: if all elements are 0, return False, if anyone is non-zero, return True
    """
    result = np.all((arr == 0))
    if result:
        return False
    else:
        return True


def region_of_interest(img, vertices):
    
    """
    Define the interest area
    :param arr: a 2D array, interest area vertices
    :return: a 2D array
    """
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def vertical_angle(img):

    angle = [500.0,500.0]
    if check_zero_in_matrix(img) == False:

        return angle

    top_points = [float("inf"),float("inf")]
    bottom_points = [float("inf"),float("inf")]
    row = img.shape[0]
    col = img.shape[1]

    bottom_array = img[[row-1],:][0]
    top_array = img[[int(2*row/3)],:][0]
    


    #top_left
    for i in range(len(top_array)):
        
        if top_array[i-1] == 255:
            top_points[0] = i-1
            break
    #top_right
    for i in range(len(top_array), -1, -1):
        if top_array[i-1] ==255:
            top_points[1] = i-1
            break
    #bottom_left 
    for i in range(len(bottom_array)):
        if bottom_array[i-1] == 255:
            bottom_points[0] = i-1
            break
    #bottom_right
    for i in range(len(bottom_array), -1, -1):
        if bottom_array[i-1] == 255:
            bottom_points[1] = i-1
            break

    #print("----------------------------------------------")
    #print("top_points", top_points)
    #print("bottom_points", bottom_points)


    bottom_centr = float(bottom_points[0] + bottom_points[1])/2.0
    diff_vertical = float(col/3)

    dis_top_points = top_points[1] - top_points[0]
    dis_bottom_points = bottom_points[1] -  bottom_points[0]

    if bottom_points[0] != float("inf") and bottom_points[1] != float("inf") and  top_points[0] != float("inf") and top_points[1] != float("inf") and dis_bottom_points != 0:
        if top_points[1] < bottom_centr:
            angle[0] = (-1)*math.degrees(math.atan(float(abs(top_points[0]- bottom_centr)/diff_vertical)))
            angle[1] = (-1)*math.degrees(math.atan(float(abs(top_points[1]- bottom_centr)/diff_vertical)))
        elif top_points[0] > bottom_centr:
            angle[0] = math.degrees(math.atan(float(abs(top_points[0]- bottom_centr)/diff_vertical)))
            angle[1] = math.degrees(math.atan(float(abs(top_points[1]- bottom_centr)/diff_vertical)))
        else:
            angle[0] = (-1)*math.degrees(math.atan(float(abs(top_points[0]- bottom_centr)/diff_vertical)))
            angle[1] = math.degrees(math.atan(float(abs(top_points[1]- bottom_centr)/diff_vertical)))


    return angle

def keep_centr(img,points_of_interest):
    row = img.shape[0]
    col = img.shape[1]

    line = row/6
    bottom_array = img[[row - 1],:][0]

    bottom_points_index = [-1,-1]
 
    #bottom_left 
    for i in range(len(bottom_array)):
        if bottom_array[i-1] != 0:
            #print("first non_0")
            bottom_points_index[0] = i-1
    #bottom_right
    for i in range(len(bottom_array), -1, -1):
        if bottom_array[i-1] != 0:
            #print("second non_0")
            bottom_points_index[1] = i-1


    centr_point  =  float(bottom_points_index[0]+ bottom_points_index[1])/2.0
   
    if centr_point < 3.0*float(col)/8.0:
        return -1.0
    if centr_point > 5.0*float(col)/8.0:
        return 1.0
    return 0.0


def process(image):
    row = image.shape[0]
    col = image.shape[1]
    
    region_of_interest_vertices_centr_botton = [
        (0, row),
        (0, row/3),
        (col, row/3),
        (col-1, row-1)
    
    ]

    
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_b = np.array([17, 92, 137])
    u_b = np.array([60, 255, 255])
    mask = cv2.inRange(hsv_frame, l_b, u_b)
    cv2.imshow("Frame", image)
    cv2.resizeWindow('Frame', (640, 480)

    res = cv2.bitwise_and(image, image, mask= mask)
    gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    
    canny_image = cv2.Canny(gray_image, 100, 200)
    
    cropped_image_centr_botton = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices_centr_botton], np.int32))

  
    if check_zero_in_matrix(cropped_image_centr_botton):
        check_road = 1.0
    else:
        check_road = 0.0

    centr_check = 0.0
    if check_road == 1.0:
        angle_list= vertical_angle(cropped_image_centr_botton)
        centr_check = keep_centr(cropped_image_centr_botton, region_of_interest_vertices_centr_botton) 
    else:
        angle_list = [500.0,500.0]


    return check_road, centr_check, round(angle_list[0],1), round(angle_list[1],1)

def talker():
    data_publisher = rospy.Publisher('cam_values', Float32MultiArray, queue_size = 20)
    rospy.init_node('talker', anonymous = True)
    rate_1 = rospy.Rate(60)
    cap = cv2.VideoCapture(0)
    a = Detector()
    flag = 0
    stop = 0.0
    right = 0.0
    while not rospy.is_shutdown():
        while cap.isOpened():
            ret, frame = cap.read()
            msg = Float32MultiArray()
            mag_temp = process(frame)
            frame_tran = cv2.resize(frame,(640, 480),fx = 0,fy = 0,interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame_tran, cv2.COLOR_BGR2GRAY)
            frame_tran[:,:,0] = gray
            frame_tran[:,:,1] = gray
            frame_tran[:,:,2] = gray
            x,y,z = a.pred(frame_tran) #x:classes, y:scores
            mag_temp = list(mag_temp)
            if flag <= 180 and flag > 0  :
                flag = flag - 1
                stop = 0.0
            elif flag > 180 :
                flag = flag - 1
                stop = 1.0
            if flag == 0:
                if x[0] == 1.0 and y[0] > 0.7:
                    stop = 1.0
                    flag = 360

            mag_temp.extend([stop,right])
            msg.data = mag_temp
            rospy.loginfo(msg)
            data_publisher.publish(msg)
            rate_1.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
