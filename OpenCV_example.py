import cv2
import numpy as np
from adafruit_servokit import ServoKit
import rospy


kit=ServoKit(channels=16)

def check_zero_in_matrix(arr):
    """
    Check if all 2D numpy array contains only 0
    :param arr: a 2D array
    :return: if all elements are 0, return 0, if anyone is non-zero, return 1
    """
    result = np.all((arr == 0))
    if result:
        return 0
    else:
        return 1


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

image = cv2.imread('turn_right.JPG')
height = image.shape[0]
wide = image.shape[1]
region_of_interest_vertices_centr_up = [
    (300, 0),
    (300,300),
    (600, 300),
    (600,0)

]
region_of_interest_vertices_left_up = [
    (0, 0),
    (0,200),
    (200, 200),
    (200,0)
]
region_of_interest_vertices_right_up = [
    (wide - 200, 0),
    (wide - 200, 200),
    (wide, 0),
    (wide, 200)
]

hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
l_b = np.array([17, 92, 137])
u_b = np.array([36, 255, 255])
mask = cv2.inRange(hsv_frame, l_b, u_b)
res = cv2.bitwise_and(image, image, mask= mask)
gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

canny_image = cv2.Canny(gray_image, 100, 200)

cropped_image_centr_up = region_of_interest(canny_image,
                np.array([region_of_interest_vertices_centr_up], np.int32))

cropped_image_right_up = region_of_interest(canny_image,
                np.array([region_of_interest_vertices_right_up], np.int32))

cropped_image_left_up = region_of_interest(canny_image,
                np.array([region_of_interest_vertices_left_up], np.int32))



#cv2.imshow('cropped_image_right_up', cropped_image_right_up)
#cv2.imshow('cropped_image_left_up', cropped_image_left_up)
#cv2.imshow('cropped_image_centr_up', cropped_image_centr_up)


if check_zero_in_matrix(cropped_image_centr_up) == 0:
    if check_zero_in_matrix(cropped_image_left_up) == 0:
        if check_zero_in_matrix(cropped_image_right_up) == 0:
            kit.continuous_servo[11].throttle = 0
        else:
            kit.continuous_servo[0].angle = 135
    else:
        kit.continuous_servo[0].angle = 135


cv2.waitKey(0)


def listener():
    rospy.init_node('scan_values')
    #rospy.Subscriber('/scan', LaserScan, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()