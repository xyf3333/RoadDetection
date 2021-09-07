#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 09:52:57 2021

@author: py
"""
import cv2
import numpy as np
import rospy
from std_msgs.msg import Int32MultiArray


def check_zero_in_matrix(arr):
    """
    Check if all 2D numpy array contains only 0
    :param arr: a 2D array
    :return: if all elements are 0, return 0, if any element is non-zero, return 1
    """
    result = np.all((arr == 0))
    if result:
        return 0
    else:
        return 1


def region_of_interest(img, vertices):

    """
    Cropping the image using vertices
    :param arr: image, vertices of interest area
    :return: cropped image only represent interest region
    """

    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    cropped_image = cv2.bitwise_and(img, mask)
    return cropped_image



def process(image):

    """
    processing image data
    :param arr: image
    :return: if all elements are 0, return 0, if any element is non-zero, return 1
    """


    wide = image.shape[0]
    height = image.shape[1]

    # definite the interest area
    region_of_interest_vertices_centr_botton = [
        (wide/8, height ),
        (wide/8,height/3),
        (7*wide/8, height/3),
        (7*wide/8, height)
    
    ]
    region_of_interest_vertices_left = [
        (0, height),
        (0,height/3),
        (wide/3, height/3),
        (wide/3,height)
    ]
    region_of_interest_vertices_right = [
        (2*wide/3, height),
        (2*wide/3, height/3),
        (wide, height/3),
        (wide, height)
    ]

    #image data conversion

    #detect yellow road trail using HSV color boundary
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_b = np.array([17, 92, 137])
    u_b = np.array([50, 255, 255])
    mask = cv2.inRange(hsv_frame, l_b, u_b)
    res = cv2.bitwise_and(image, image, mask= mask)

    #transfer to gray image
    gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    #detect edge of road
    canny_image = cv2.Canny(gray_image, 100, 200)

    cropped_image_centr_botton = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices_centr_botton], np.int32))
    
    cropped_image_right = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices_right], np.int32))
    
    cropped_image_left = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices_left], np.int32))


    centr = check_zero_in_matrix(cropped_image_centr_botton)
    left_sum = cropped_image_left.sum()                                 #the sum of pixel value
    right_sum = cropped_image_right.sum()                               #the sum of pixel value

    return [left_sum, centr, right_sum]

def talker():

    rospy.init_node('talker', anonymous=True)                                               #initial publisher node
    data_publisher = rospy.Publisher('cam_values', Int32MultiArray, queue_size= 10)         #define the publisher data
    rate = rospy.Rate(10)
    cap = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow('vv',frame)
            msg = Int32MultiArray()
            msg.data = process(frame) 
            rospy.loginfo(msg)
            data_publisher.publish(msg)
            rate.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    