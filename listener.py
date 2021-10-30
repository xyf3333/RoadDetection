#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:23:44 2021

@author: py
"""


from adafruit_servokit import ServoKit
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import message_filters
import time

            
kit  = ServoKit(channels = 16)

def callback(msg1, msg2):
    """
    INPUT: msg1: data from cam_values node, msg2: LaserScan data
    """
    check_road = msg1.data[0]
    check_centr = msg1.data[1]
    angle_1 = msg1.data[2]
    angle_2 = msg1.data[3]
    stop_sign = msg1.data[4]
    right_sign = msg1.data[5]
    left_min = min(msg2.ranges[45:90])
    top_centr_min = min(min(msg2.ranges[0:45]),min(msg2.ranges[315:359]))
    right_min = min(msg2.ranges[270:315])
    centr_2 = min(min(msg2.ranges[0:15]),min(msg2.ranges[335:359]))
    left = min(msg2.ranges[45:75])
    right =min(msg2.ranges[295:315])
    #emergency stop
    if left_min < 0.15 or right_min < 0.15 or top_centr_min < 0.25:
        kit.continuous_servo[11].throttle = 0
    #stop sign
    if stop_sign == 1.0 & centr_2 < 2.0:
        #print("stop sign detected_______________________________________________________")
        
        kit.continuous_servo[11].throttle = 0
        kit.servo[0].angle = 90
           
    else:
       
        kit.continuous_servo[11].throttle = 0.3
        kit.servo[0].angle = 90
        #avoid hit the wall
        if left_min < 0.3:
            kit.servo[0].angle = 100
        if right_min < 0.3:
            kit.servo[0].angle = 80

        if centr_2 < 2.0:
        #turn left
            if right_min < 0.25 or left > 0.6:
                kit.servo[0].angle = 45
         
        #turn right
            if  left_min < 0.25 or right > 0.6:
                kit.servo[0].angle = 135
        #stop
            if right_min < 0.3 and left_min < 0.3:
                kit.continuous_servo[11].throttle = 0  

        elif check_road == 1.0:
            if check_centr == -1.0:
                kit.servo[0].angle = 80
            elif check_centr == 1.0:
                kit.servo[0].angle = 100
            else:
                kit.servo[0].angle = 90
 
def listener():
    
    #initial the listener node
    rospy.init_node('listener', anonymous = True)
    cam_sub = message_filters.Subscriber('cam_values', Float32MultiArray)
    lidar_sub = message_filters.Subscriber('/scan', LaserScan)

    #synchronize the data from cam_sub and lidar_sub nodes
    ts = message_filters.ApproximateTimeSynchronizer([cam_sub, lidar_sub], queue_size = 10, slop = 0.1, allow_headerless = True)
    ts.registerCallback(callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
