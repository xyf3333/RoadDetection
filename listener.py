#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:23:44 2021

@author: py
"""


from adafruit_servokit import ServoKit
import rospy
from std_msgs.msg import Int32MultiA

            
kit = ServoKit(channels = 16)

def callback(data):
    """
    using data information to decide following actions.
    :param arr: a array data.
    """

    rospy.loginfo(data.data)

    left = data.data[0]
    centr = data.data[1]
    right = data.data[2]


    if centr == 0:
        kit.continuous_servo[11].throttle = 0
    else:
        kit.continuous_servo[11].throttle = 0.3

    if left > 0:
        kit.servo[0].angle = 65
    if right > 0:
        kit.servo[0].angle = 115

    if left > right:
        kit.servo[0].angle = 65
    else:
        kit.servo[0].angle = 115
    


def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('cam_values', Int32MultiArray, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()